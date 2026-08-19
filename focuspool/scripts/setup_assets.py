"""Asset setup command for FocusPool CLI."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from focuspool import REPO_ROOT, WEIGHTS_DIR


DINO_WEIGHTS_ASSET = "dinov3_vits16plus.pth"
RVT2_WEIGHTS_ASSET = "mimicgen.rvt2.ckpt"
RVT2_WEIGHTS_LOCAL_NAME = "mimicgen.rvt2_heatmap.ckpt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Set up optional FocusPool release assets. Assets must come from a "
            "GitHub release, not just a branch."
        ),
        epilog=(
            "Private release:\n"
            "  gh auth login\n"
            "  focuspool setup-assets --repo <owner/private-repo> --release-tag <tag>\n\n"
            "If the assets only exist on a dev branch and are not attached to a "
            "GitHub release, this command will not work. In that case, publish a "
            "release or place the files manually."
        ),
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repo in owner/name format.",
    )
    parser.add_argument(
        "--release-tag",
        required=True,
        help="GitHub release tag for optional FocusPool assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    parser.add_argument(
        "--skip-task-cache",
        action="store_true",
        help="Do not prebuild the CLIP task embedding cache after asset setup.",
    )
    return parser.parse_args(argv)


def download_file(url: str, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")

    print(f"[setup_assets] Downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    print(f"[setup_assets] Saved: {dst}")


def download_release_asset(
    *,
    repo: str,
    release_tag: str,
    asset_name: str,
    dst: Path,
    force: bool,
) -> bool:
    if dst.exists() and not force:
        print(f"[setup_assets] Skip existing: {dst}")
        return True

    url = f"https://github.com/{repo}/releases/download/{release_tag}/{asset_name}"
    try:
        download_file(url, dst, force=True)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    gh = shutil.which("gh")
    if gh is None:
        return False

    print(
        "[setup_assets] Direct URL returned 404; trying authenticated "
        f"download via gh for {asset_name}"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="focuspool_gh_dl_") as tmp:
        cmd = [
            gh,
            "release",
            "download",
            release_tag,
            "-R",
            repo,
            "-p",
            asset_name,
            "-D",
            tmp,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown gh error"
            raise RuntimeError(
                "Authenticated GitHub release download failed. Check that you have "
                "access to the private repo, ran 'gh auth login', and provided the "
                f"correct --repo/--release-tag. gh error: {stderr}"
            )
        src = Path(tmp) / asset_name
        if not src.exists():
            raise RuntimeError(
                f"gh reported success but asset missing in temp dir: {asset_name}"
            )
        shutil.move(str(src), dst)
        print(f"[setup_assets] Saved via gh: {dst}")
        return True


def download_optional_release_asset(
    *,
    repo: str,
    release_tag: str,
    asset_name: str,
    dst: Path,
    force: bool,
) -> None:
    try:
        downloaded = download_release_asset(
            repo=repo,
            release_tag=release_tag,
            asset_name=asset_name,
            dst=dst,
            force=force,
        )
    except RuntimeError as exc:
        print(f"[setup_assets] Optional asset skipped: {asset_name} ({exc})")
        return
    if not downloaded:
        print(f"[setup_assets] Optional asset not found, skipping: {asset_name}")


def warm_task_embedding_cache() -> None:
    """Build the task embedding cache before multi-worker jobs need it."""

    try:
        from focuspool.util.task_meta import _default_cache_path, setup_task_embedding_cache

        cache_path = _default_cache_path()
        if cache_path.exists():
            print(f"[setup_assets] Task embedding cache exists: {cache_path}")
            setup_task_embedding_cache()
        else:
            print(f"[setup_assets] Building task embedding cache: {cache_path}")
            setup_task_embedding_cache()
            print(f"[setup_assets] Task embedding cache saved: {cache_path}")
    except Exception as exc:
        print(
            "[setup_assets] WARNING: task embedding cache was not built. "
            "It will be built lazily by rerender/eval with a file lock. "
            f"Reason: {exc}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    release_base = f"https://github.com/{args.repo}/releases/download/{args.release_tag}"

    dinov3_weights = WEIGHTS_DIR / DINO_WEIGHTS_ASSET
    rvt2_weights = WEIGHTS_DIR / RVT2_WEIGHTS_LOCAL_NAME

    print(f"[setup_assets] Repo root       : {REPO_ROOT}")
    print(f"[setup_assets] Weights dir     : {WEIGHTS_DIR}")
    print(f"[setup_assets] Release source  : {release_base}")

    try:
        download_optional_release_asset(
            repo=args.repo,
            release_tag=args.release_tag,
            asset_name=DINO_WEIGHTS_ASSET,
            dst=dinov3_weights,
            force=args.force,
        )
        download_optional_release_asset(
            repo=args.repo,
            release_tag=args.release_tag,
            asset_name=RVT2_WEIGHTS_ASSET,
            dst=rvt2_weights,
            force=args.force,
        )

        if not args.skip_task_cache:
            warm_task_embedding_cache()
    except Exception as exc:
        print(f"[setup_assets] Failed: {exc}", file=sys.stderr)
        return 1

    print("[setup_assets] Done.")
    return 0
