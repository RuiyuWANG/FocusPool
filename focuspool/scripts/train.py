"""Train command for FocusPool CLI."""

from __future__ import annotations

import argparse
import sys

import hydra
import torch
from omegaconf import OmegaConf

from ..workspace.base_workspace import BaseWorkspace


MAX_STEPS = {
    "square_d0": 400,
    "stack_d1": 400,
    "stack_three_d1": 400,
    "square_d2": 400,
    "threading_d2": 400,
    "coffee_d2": 400,
    "three_piece_assembly_d2": 500,
    "hammer_cleanup_d1": 500,
    "mug_cleanup_d1": 500,
    "kitchen_d1": 800,
    "nut_assembly_d0": 500,
    "pick_place_d0": 1000,
    "coffee_preparation_d1": 800,
    "tool_hang": 700,
    "can": 400,
    "lift": 400,
    "square": 400,
}

HYDRA_OPTION_FLAGS = {
    "--help",
    "--hydra-help",
    "--version",
    "--cfg",
    "--resolve",
    "--package",
    "--run",
    "--multirun",
    "--shell-completion",
    "--config-path",
    "--config-name",
    "--config-dir",
    "--experimental-rerun",
    "--info",
}


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _attn_prior_suffix(enabled, weight, sigma, decay_steps) -> str:
    if not _truthy(enabled):
        return ""
    decay = int(decay_steps)
    decay_part = f"_decay_{_compact_steps(decay)}" if decay > 0 else ""
    return f"_attnprior-w{float(weight):g}-sig{float(sigma):g}{decay_part}"


def _compact_steps(steps: int) -> str:
    if steps % 1000 == 0:
        return f"{steps // 1000}k"
    if steps > 1000:
        return f"{steps / 1000:g}k"
    return str(steps)


def _register_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "add_int", lambda a, b: int(a) + int(b), replace=True
    )
    OmegaConf.register_new_resolver(
        "divide", lambda a, b: float(a) / float(b), replace=True
    )
    OmegaConf.register_new_resolver(
        "get_max_steps", lambda task_name: MAX_STEPS.get(task_name, 800), replace=True
    )
    OmegaConf.register_new_resolver(
        "action_name",
        lambda action_rep: {"absolute": "abs", "delta": "rel"}.get(
            str(action_rep), str(action_rep)
        ),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "focus_refine_iters_suffix",
        lambda pooling, iters: (
            f"_iters-{int(iters)}" if str(pooling) == "focus_refine" else ""
        ),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "attn_prior_suffix",
        _attn_prior_suffix,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "overlay_suffix",
        lambda enabled: "_overlay" if _truthy(enabled) else "",
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "no_eih_suffix",
        lambda enabled: "" if _truthy(enabled) else "_no_eih",
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "no_eih_tag",
        lambda enabled: "" if _truthy(enabled) else "no_eih",
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "overlay_prob",
        lambda enabled, prob: float(prob) if _truthy(enabled) else 0.0,
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "stage_pooled_attn_prior",
        lambda enabled, pooling: _truthy(enabled) and str(pooling) == "focus_refine",
        replace=True,
    )


def _enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _normalize_train_argv(argv: list[str]) -> list[str]:
    """Translate friendly CLI flags into Hydra overrides.

    Example:
      --task-name three_piece_assembly_d2 -> task_name=three_piece_assembly_d2
      --n-demo=100 -> n_demo=100
    """
    normalized: list[str] = []
    idx = 0

    while idx < len(argv):
        token = argv[idx]

        if token == "-h":
            normalized.append(token)
            idx += 1
            continue

        if token.startswith("--"):
            option, sep, value = token.partition("=")
            if option in HYDRA_OPTION_FLAGS:
                normalized.append(token)
                hydra_option_takes_value = option in {
                    "--cfg",
                    "--package",
                    "--config-path",
                    "--config-name",
                    "--config-dir",
                    "--experimental-rerun",
                    "--info",
                }
                if not sep and hydra_option_takes_value and idx + 1 < len(argv):
                    normalized.append(argv[idx + 1])
                    idx += 2
                    continue
                idx += 1
                continue

            key = option[2:].replace("-", "_")
            if sep:
                normalized.append(f"{key}={value}")
                idx += 1
                continue

            if idx + 1 < len(argv) and not argv[idx + 1].startswith("-"):
                normalized.append(f"{key}={argv[idx + 1]}")
                idx += 2
                continue

            normalized.append(f"{key}=true")
            idx += 1
            continue

        normalized.append(token)
        idx += 1

    return normalized


@hydra.main(
    version_base=None,
    config_path="../config",
)
def _hydra_main(cfg: OmegaConf) -> None:
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


def main(argv: list[str] | None = None) -> None:
    # Keep logs streaming in long-running training jobs.
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

    _register_resolvers()
    _enable_tf32()

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    normalized_argv = _normalize_train_argv(raw_argv)

    parser = argparse.ArgumentParser(
        description="Train models with Hydra config overrides.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Hydra config name to train with, e.g. train_stage_pooled_policy.",
    )
    args, hydra_args = parser.parse_known_args(normalized_argv)

    sys.argv = ["focuspool train"]
    if args.config_name:
        sys.argv.append(f"--config-name={args.config_name}")
    sys.argv.extend(hydra_args)

    _hydra_main()


if __name__ == "__main__":
    main()
