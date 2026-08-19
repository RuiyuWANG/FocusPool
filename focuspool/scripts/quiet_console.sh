#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---apply}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEPS_ROOT="${FOCUSPOOL_SUITE_DEPS_ROOT:-"$ROOT/../focuspool-suite-deps"}"
STACK_DIR="${FOCUSPOOL_SUITE_STACK_DIR:-mimic}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python)}"

"$PYTHON" - "$MODE" "$DEPS_ROOT/$STACK_DIR" <<'PY'
from pathlib import Path
import sys

mode = sys.argv[1]
suite_root = Path(sys.argv[2])

if mode not in {"--apply", "--restore"}:
    raise SystemExit("usage: focuspool/scripts/quiet_console.sh [--apply|--restore]")


def replace_block(rel_path: str, verbose: str, quiet: str) -> None:
    path = suite_root / rel_path
    if not path.exists():
        raise RuntimeError(f"missing file: {path}")
    text = path.read_text()
    src, dst = (verbose, quiet) if mode == "--apply" else (quiet, verbose)
    if dst in text:
        print(f"[quiet-console] already {'quiet' if mode == '--apply' else 'restored'}: {rel_path}")
        return
    if src not in text:
        raise RuntimeError(f"expected block not found in {rel_path}")
    path.write_text(text.replace(src, dst))
    print(f"[quiet-console] {'quieted' if mode == '--apply' else 'restored'}: {rel_path}")


replace_block(
    "robosuite/robosuite/macros.py",
    """# Override with macros from macros_private.py file, if it exists
try:
    from robosuite.macros_private import *
except ImportError:
    import robosuite
    from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

    ROBOSUITE_DEFAULT_LOGGER.warn("No private macro file found!")
    ROBOSUITE_DEFAULT_LOGGER.warn("It is recommended to use a private macro file")
    ROBOSUITE_DEFAULT_LOGGER.warn("To setup, run: python {}/scripts/setup_macros.py".format(robosuite.__path__[0]))
""",
    """# Override with macros from macros_private.py file, if it exists
# try:
#     from robosuite.macros_private import *
# except ImportError:
#     import robosuite
#     from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

#     ROBOSUITE_DEFAULT_LOGGER.warn("No private macro file found!")
#     ROBOSUITE_DEFAULT_LOGGER.warn("It is recommended to use a private macro file")
#     ROBOSUITE_DEFAULT_LOGGER.warn("To setup, run: python {}/scripts/setup_macros.py".format(robosuite.__path__[0]))
""",
)

replace_block(
    "robosuite/robosuite/wrappers/__init__.py",
    """try:
    from robosuite.wrappers.gym_wrapper import GymWrapper
except:
    print("Warning: make sure gym is installed if you want to use the GymWrapper.")
""",
    """# try:
#     from robosuite.wrappers.gym_wrapper import GymWrapper
# except:
#     print("Warning: make sure gym is installed if you want to use the GymWrapper.")
""",
)

replace_block(
    "robomimic/robomimic/macros.py",
    """try:
    from robomimic.macros_private import *
except ImportError:
    from robomimic.utils.log_utils import log_warning
    import robomimic
    log_warning(
        "No private macro file found!"\\
        "\\nIt is recommended to use a private macro file"\\
        "\\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
    )
""",
    """# try:
#     from robomimic.macros_private import *
# except ImportError:
#     from robomimic.utils.log_utils import log_warning
#     import robomimic
#     log_warning(
#         "No private macro file found!"\\
#         "\\nIt is recommended to use a private macro file"\\
#         "\\nTo setup, run: python {}/scripts/setup_macros.py".format(robomimic.__path__[0])
#     )
""",
)

replace_block(
    "robomimic/robomimic/utils/env_utils.py",
    """    print("Created environment with name {}".format(env_name))
    print("Action size is {}".format(env.action_dimension))
""",
    """    # print("Created environment with name {}".format(env_name))
    # print("Action size is {}".format(env.action_dimension))
""",
)

replace_block(
    "robomimic/robomimic/utils/obs_utils.py",
    """    print("\\n============= Initialized Observation Utils with Obs Spec =============\\n")
    for obs_modality, obs_keys in OBS_MODALITIES_TO_KEYS.items():
        print("using obs modality: {} with keys: {}".format(obs_modality, obs_keys))
""",
    """    # print("\\n============= Initialized Observation Utils with Obs Spec =============\\n")
    # for obs_modality, obs_keys in OBS_MODALITIES_TO_KEYS.items():
    #     print("using obs modality: {} with keys: {}".format(obs_modality, obs_keys))
""",
)

replace_block(
    "mimicgen/mimicgen/__init__.py",
    """try:
    from mimicgen.envs.robosuite.hammer_cleanup import *
    from mimicgen.envs.robosuite.kitchen import *
except ImportError as e:
    print("WARNING: robosuite task zoo environments not imported, possibly because robosuite_task_zoo is not installed...")
    print("Got error: {}".format(e))
""",
    """# try:
#     from mimicgen.envs.robosuite.hammer_cleanup import *
#     from mimicgen.envs.robosuite.kitchen import *
# except ImportError as e:
#     print("WARNING: robosuite task zoo environments not imported, possibly because robosuite_task_zoo is not installed...")
#     print("Got error: {}".format(e))
""",
)
PY
