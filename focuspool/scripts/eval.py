"""Evaluation command for FocusPool CLI."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pathlib
import pprint
import random
import shutil
import sys

import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf.omegaconf import open_dict

from ..util.formatting import pretty_print_nested


EVAL_DEVICE = "cuda:0"


def _configure_line_buffering() -> None:
    # Keep logs streaming in long-running evaluation jobs.
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _rename_eval_media_dir(output_dir: str, test_start_seed: int) -> tuple[str, str] | None:
    source_dir = os.path.join(output_dir, "media")
    target_dir = os.path.join(output_dir, f"media_seed_{int(test_start_seed)}")

    if not os.path.isdir(source_dir) or source_dir == target_dir:
        return None

    if os.path.exists(target_dir):
        for entry in os.listdir(source_dir):
            shutil.move(os.path.join(source_dir, entry), os.path.join(target_dir, entry))
        os.rmdir(source_dir)
    else:
        os.replace(source_dir, target_dir)

    return source_dir, target_dir


def _rewrite_media_path(path: str, renamed_media_dir: tuple[str, str] | None) -> str:
    if renamed_media_dir is None:
        return path

    source_dir, target_dir = renamed_media_dir
    if path == source_dir:
        return target_dir
    if path.startswith(source_dir + os.sep):
        return target_dir + path[len(source_dir) :]
    return path


def _load_checkpoint_payload(checkpoint: str) -> dict:
    with open(checkpoint, "rb") as f:
        return torch.load(f, pickle_module=dill)


def _checkpoint_paths(checkpoint: str) -> list[str]:
    if os.path.isfile(checkpoint):
        return [checkpoint]
    if os.path.isdir(checkpoint):
        return sorted(
            os.path.join(checkpoint, file_name)
            for file_name in os.listdir(checkpoint)
            if file_name.endswith(".ckpt")
        )
    raise ValueError(f"Invalid checkpoint path: {checkpoint}")


def _checkpoint_output_dir(ckpt_path: str, output_dir: str | None) -> str:
    if output_dir is not None:
        return output_dir

    ckpt_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    return os.path.join(os.path.dirname(ckpt_dir), "eval", ckpt_name)


def evaluate_checkpoint(
    checkpoint: str,
    output_dir: str,
    device: str,
    n_eval_rollouts: int,
    seed: int,
    test_start_seed: int,
    env_name: str | None,
    n_envs: int,
    shuffle_table_texture: bool,
    payload: dict | None = None,
) -> None:
    if payload is None:
        payload = _load_checkpoint_payload(checkpoint)
    cfg = payload["cfg"]

    with open_dict(cfg):
        cfg.training.seed = seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if getattr(cfg.training, "use_ema", False):
        policy = workspace.ema_model

    device_obj = torch.device(device)
    policy.to(device_obj)
    policy.eval()

    with open_dict(cfg):
        for key in ("n_train", "n_train_vis", "train_start_idx"):
            if key in cfg.task.env_runner:
                del cfg.task.env_runner[key]
        cfg.task.env_runner.n_test = n_eval_rollouts
        cfg.task.env_runner.n_envs = n_envs
        cfg.task.env_runner.n_test_vis = 10
        cfg.task.env_runner.test_start_seed = test_start_seed
        if env_name is not None:
            cfg.task.env_runner.env_name = env_name

    runtime_cfg = {}
    if hasattr(policy, "get_runtime_config"):
        runtime_cfg = copy.deepcopy(policy.get_runtime_config())
    runtime_cfg["Evaluation"] = {
        "Checkpoint": checkpoint,
        "Output Dir": output_dir,
        "Device": device,
        "Environment": cfg.task.env_runner.env_name,
        "Rollouts": n_eval_rollouts,
        "Num Envs": n_envs,
        "Main Process Seed": seed,
        "Test Start Seed": cfg.task.env_runner.test_start_seed,
        "Shuffle Table Texture": "Enabled" if shuffle_table_texture else "Disabled",
    }
    pretty_print_nested(
        runtime_cfg,
        title="Evaluation Configuration",
        pad_before=True,
        pad_after=True,
    )

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        shuffle_table_texture=shuffle_table_texture,
    )
    runner_log = env_runner.run(policy)
    renamed_media_dir = _rename_eval_media_dir(output_dir, cfg.task.env_runner.test_start_seed)

    json_log = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = _rewrite_media_path(value._path, renamed_media_dir)
        else:
            json_log[key] = value

    eval_env_name = cfg.task.env_runner.env_name if env_name is None else env_name

    detailed_log = {
        "Mean Score": json_log.get("test/mean_score", "Not found"),
        "Test Environment": eval_env_name,
        "Number of Rollouts": n_eval_rollouts,
        "Random Seeds": {
            "main_process_seed": seed,
            "test_start_seed": cfg.task.env_runner.test_start_seed,
        },
        "Paths": {
            "Experiment Folder": os.path.dirname(checkpoint),
            "Checkpoint File": os.path.basename(checkpoint),
            "Output Directory": output_dir,
            "Media Directory": (
                os.path.join(output_dir, f"media_seed_{cfg.task.env_runner.test_start_seed}")
                if renamed_media_dir is not None
                else os.path.join(output_dir, "media")
            ),
        },
    }

    pprint.pprint(detailed_log, indent=2)

    with open(os.path.join(output_dir, "eval_log.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_log, f, indent=2, sort_keys=True)

    with open(
        os.path.join(output_dir, "raw_runner_log.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(json_log, f, indent=2, sort_keys=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FocusPool checkpoint(s).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or directory")
    parser.add_argument("--env-name", dest="env_name", default=None)
    parser.add_argument(
        "--n-eval-rollouts",
        dest="n_eval_rollouts",
        type=int,
        default=50,
    )
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Main-process RNG seed override. Defaults to cfg.training.seed from the checkpoint.",
    )
    parser.add_argument(
        "--test-start-seed",
        dest="test_start_seed",
        type=int,
        default=None,
        help=(
            "Evaluation rollout seed override. Defaults to "
            "cfg.task.env_runner.test_start_seed from the checkpoint."
        ),
    )
    parser.add_argument("--n-envs", dest="n_envs", type=int, default=50)
    parser.add_argument(
        "--shuffle-table-texture",
        dest="shuffle_table_texture",
        action="store_true",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    _configure_line_buffering()
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("eval requires a CUDA GPU")

    ckpt_list = _checkpoint_paths(args.checkpoint)

    print(f"Found {len(ckpt_list)} checkpoints to evaluate.")

    for ckpt_path in ckpt_list:
        ckpt_output_dir = _checkpoint_output_dir(ckpt_path, args.output_dir)
        pathlib.Path(ckpt_output_dir).mkdir(parents=True, exist_ok=True)

        payload = _load_checkpoint_payload(ckpt_path)
        cfg = payload["cfg"]
        seed = cfg.training.seed if args.seed is None else args.seed
        test_start_seed = (
            cfg.task.env_runner.test_start_seed
            if args.test_start_seed is None
            else args.test_start_seed
        )
        evaluate_checkpoint(
            checkpoint=ckpt_path,
            output_dir=ckpt_output_dir,
            device=EVAL_DEVICE,
            n_eval_rollouts=args.n_eval_rollouts,
            seed=seed,
            test_start_seed=test_start_seed,
            env_name=args.env_name,
            n_envs=args.n_envs,
            shuffle_table_texture=args.shuffle_table_texture,
            payload=payload,
        )
