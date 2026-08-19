#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUITE_RESOURCE_DIR="$ROOT/.dep"
DEPS_ROOT="${1:-"$ROOT/../focuspool-suite-deps"}"
STACK="${FOCUSPOOL_SUITE_STACK:-mimicgen}"
EXPECTED_CONDA_ENV="${FOCUSPOOL_CONDA_ENV:-focuspool}"

if [[ -n "$EXPECTED_CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$EXPECTED_CONDA_ENV" ]]; then
  echo "Expected conda environment '$EXPECTED_CONDA_ENV', but active environment is '${CONDA_DEFAULT_ENV:-none}'." >&2
  echo "Run 'conda activate $EXPECTED_CONDA_ENV' first, or set FOCUSPOOL_CONDA_ENV to the target env name." >&2
  echo "Set FOCUSPOOL_CONDA_ENV= to skip this check." >&2
  exit 1
fi

case "$STACK" in
  mimic | mimicgen)
    STACK="mimicgen"
    STACK_DIR="mimic"
    ;;
  *)
    echo "Unsupported suite stack '$STACK'. This branch ships only mimicgen." >&2
    exit 1
    ;;
esac

LOCK_FILE="${FOCUSPOOL_SUITE_LOCK:-"$SUITE_RESOURCE_DIR/mimicgen.lock"}"

if [[ ! -f "$LOCK_FILE" ]]; then
  echo "Missing suite dependency lock file: $LOCK_FILE" >&2
  exit 1
fi

mkdir -p "$DEPS_ROOT"
echo "Installing $STACK suite dependencies from $LOCK_FILE into ${CONDA_DEFAULT_ENV:-the active Python environment}."
echo "Checkout directory: $DEPS_ROOT/$STACK_DIR"

tracked_changes() {
  git -C "$1" status --porcelain --untracked-files=no
}

log_step() {
  printf '[suite] %s\n' "$*"
}

install_editable() {
  local name="$1"
  local dst="$2"

  local pip_args=(--no-input -e "$dst")
  # The conda environment pins the MimicGen Python dependencies. Avoid letting
  # pip's resolver/network work make the final editable install appear stuck.
  if [[ "$name" == "mimicgen" ]]; then
    pip_args=(--no-input --no-deps -e "$dst")
  fi

  PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
    python -m pip install "${pip_args[@]}" < /dev/null
}

ensure_robosuite_macros_private() {
  local dst="$1"
  local package_dir="$dst/robosuite"
  local macros_file="$package_dir/macros.py"
  local private_file="$package_dir/macros_private.py"

  if [[ ! -f "$macros_file" ]]; then
    echo "Warning: robosuite macros.py not found at $macros_file; skipping macros_private.py setup." >&2
    return
  fi

  printf '%s\n' \
    '# Created by focuspool/scripts/setup_suite_deps.sh.' \
    '# Keep numba cache disabled for editable external checkouts.' \
    'CACHE_NUMBA = False' \
    > "$private_file"
  echo "Configured robosuite macros_private.py: $private_file"
}

while read -r name url sha patch; do
  [[ -z "${name:-}" || "$name" == \#* ]] && continue

  dst="$DEPS_ROOT/$STACK_DIR/$name"
  log_step "$name: preparing checkout at $dst"
  if [[ ! -d "$dst/.git" ]]; then
    log_step "$name: cloning $url"
    git clone "$url" "$dst"
  fi

  current="$(git -C "$dst" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$current" != "$sha" ]]; then
    if [[ -n "$(tracked_changes "$dst")" ]]; then
      echo "Refusing to checkout $name: $dst has local changes." >&2
      exit 1
    fi
    log_step "$name: fetching tags"
    git -C "$dst" fetch --tags origin
    log_step "$name: checking out $sha"
    git -C "$dst" checkout "$sha"
  fi

  if [[ "$patch" == "-" ]]; then
    if [[ -n "$(tracked_changes "$dst")" ]]; then
      echo "Refusing to use $name: $dst has local changes but the lock expects no patch." >&2
      echo "Reset or recreate this checkout before re-running setup." >&2
      exit 1
    fi
  else
    if [[ "$patch" = /* ]]; then
      patch_path="$patch"
    else
      patch_path="$(cd "$(dirname "$LOCK_FILE")" && pwd)/$patch"
    fi
    log_step "$name: checking patch $patch_path"
    if git -C "$dst" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
      log_step "$name: patch already present, normalizing local patch state"
      git -C "$dst" apply --reverse "$patch_path"
      if git -C "$dst" diff --quiet --; then
        git -C "$dst" apply "$patch_path"
      else
        git -C "$dst" apply "$patch_path"
        echo "Patch subset already applied, but $dst has extra local changes." >&2
        echo "Reset or recreate this checkout before re-running setup if you want the minimal patch set." >&2
        exit 1
      fi
      echo "Patch already applied exactly: $patch"
    else
      if [[ -n "$(tracked_changes "$dst")" ]]; then
        echo "Refusing to apply $patch: $dst has local changes." >&2
        exit 1
      fi
      log_step "$name: applying patch $patch"
      git -C "$dst" apply --check "$patch_path"
      git -C "$dst" apply "$patch_path"
    fi
  fi

  if [[ "$name" == "robosuite" && -d "$dst/robosuite" ]]; then
    ensure_robosuite_macros_private "$dst"
  fi
  log_step "$name: installing editable package"
  install_editable "$name" "$dst"
  log_step "$name: done"
done < "$LOCK_FILE"
