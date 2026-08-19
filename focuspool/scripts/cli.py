"""Main command dispatcher for FocusPool CLI."""

from __future__ import annotations

import importlib
import sys

HELP_TEXT = """FocusPool CLI

Usage:
  focuspool <command> [args]
  python -m focuspool.scripts.cli <command> [args]

Commands:
  train              Run training (Hydra args pass-through)
  eval               Evaluate one checkpoint or a checkpoint directory
  rerender-dataset   Convert raw HDF5 demos into LMDB caches
  playback-dataset   Playback cached observations or open-loop absolute actions
  merge-datasets     Merge per-task LMDB caches for multitask training
  setup-assets       Download optional release assets

Run 'focuspool <command> --help' for command-specific options.
"""

SCRIPT_COMMANDS = {
    "eval": "eval",
    "rerender-dataset": "rerender_dataset",
    "playback-dataset": "playback_dataset",
    "merge-datasets": "merge_lmdb_caches",
}


def _run_script_command(command: str, module_name: str, argv: list[str]) -> int:
    """Run a script-style command whose main() reads sys.argv."""
    try:
        module = importlib.import_module(f".{module_name}", package=__package__)
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency for '{command}': {exc.name}. "
            "Install project dependencies first."
        )
        return 1

    sys.argv = [f"focuspool {command}", *argv]
    module.main()
    return 0


def _run_train_command(argv: list[str]) -> int:
    try:
        from . import train as train_cmd
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency for 'train': {exc.name}. "
            "Install project dependencies first."
        )
        return 1

    train_cmd.main(argv)
    return 0


def _run_setup_assets_command(argv: list[str]) -> int:
    from . import setup_assets as setup_assets_cmd

    return setup_assets_cmd.main(argv)


LOCAL_COMMANDS = {
    "train": _run_train_command,
    "setup-assets": _run_setup_assets_command,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(HELP_TEXT.strip())
        return 0

    command = args[0]
    rest = args[1:]

    local_command = LOCAL_COMMANDS.get(command)
    if local_command is not None:
        return local_command(rest)

    script_module = SCRIPT_COMMANDS.get(command)
    if script_module is not None:
        return _run_script_command(command, script_module, rest)

    print(f"Unknown command: {command}\n")
    print(HELP_TEXT.strip())
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
