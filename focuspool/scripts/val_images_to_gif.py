"""Convert saved validation visualization images into an animated GIF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image
from PIL import ImageDraw


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
STEP_RE = re.compile(r"(?:^|[_-])step[_-]?(\d+)|train_step_(\d+)_viz")


def _resolve_image_dir(path: Path) -> Path:
    path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"input path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"input path must be a directory: {path}")

    if path.name == "val":
        return path

    candidate = path / "visualization" / "val"
    if candidate.is_dir():
        return candidate

    return path


def _sort_key(path: Path) -> tuple[int, int, str]:
    step = _step_from_path(path)
    if step is None:
        return (1, 0, path.name)
    return (0, step, path.name)


def _step_from_path(path: Path) -> int | None:
    match = STEP_RE.search(path.stem)
    if match is None:
        return None
    return next(int(group) for group in match.groups() if group is not None)


def _find_images(image_dir: Path, pattern: str) -> list[Path]:
    images = [
        path
        for path in image_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=_sort_key)


def _load_frame(path: Path, max_width: int | None) -> Image.Image:
    image = Image.open(path)
    image.load()
    frame = image.convert("RGB")

    if max_width is not None and max_width > 0 and frame.width > max_width:
        height = round(frame.height * (max_width / frame.width))
        frame = frame.resize((max_width, height), Image.Resampling.LANCZOS)

    return frame


def _pad_to_size(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    if frame.size == size:
        return frame
    canvas = Image.new("RGB", size, color=(0, 0, 0))
    x = (size[0] - frame.width) // 2
    y = (size[1] - frame.height) // 2
    canvas.paste(frame, (x, y))
    return canvas


def _text_size(draw: ImageDraw.ImageDraw, text: str) -> tuple[int, int]:
    if hasattr(draw, "textbbox"):
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text)
            return right - left, bottom - top
        except ValueError:
            pass
    return draw.textsize(text)


def _draw_frame_label(frame: Image.Image, *, frame_idx: int, source_path: Path) -> None:
    step = _step_from_path(source_path)
    label = f"idx {frame_idx:04d}" if step is None else f"idx {frame_idx:04d}  step {step}"
    draw = ImageDraw.Draw(frame)
    text_w, text_h = _text_size(draw, label)
    pad = 5
    x0, y0 = 6, 6
    draw.rectangle(
        [x0 - pad, y0 - pad, x0 + text_w + pad, y0 + text_h + pad],
        fill=(0, 0, 0),
    )
    draw.text((x0, y0), label, fill=(255, 255, 255))


def _draw_column_labels(frame: Image.Image, *, frame_idx: int, columns: int) -> None:
    columns = int(columns)
    if columns <= 0:
        return
    draw = ImageDraw.Draw(frame)
    col_width = frame.width / columns
    pad_x = 4
    pad_y = 3

    for col_idx in range(columns):
        label = f"{frame_idx:03d}"
        text_w, text_h = _text_size(draw, label)
        col_right = round((col_idx + 1) * col_width)
        x0 = col_right - text_w - 6
        y0 = frame.height - text_h - 6
        col_left = round(col_idx * col_width)
        x0 = max(col_left + 2, min(x0, frame.width - text_w - 2))
        y0 = max(2, min(y0, frame.height - text_h - 2))
        draw.rectangle(
            [
                x0 - pad_x,
                y0 - pad_y,
                x0 + text_w + pad_x,
                y0 + text_h + pad_y,
            ],
            fill=(0, 0, 0),
        )
        draw.text((x0, y0), label, fill=(255, 255, 255))


def make_gif(
    input_dir: Path,
    output_path: Path | None,
    *,
    fps: float,
    pattern: str,
    max_width: int | None,
    frame_index: bool,
    column_index: bool,
    columns: int,
) -> Path:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    image_dir = _resolve_image_dir(input_dir)
    image_paths = _find_images(image_dir, pattern)
    if not image_paths:
        raise FileNotFoundError(
            f"no images found in {image_dir} matching {pattern!r}; "
            f"supported extensions: {sorted(IMAGE_EXTENSIONS)}"
        )

    frames = [_load_frame(path, max_width=max_width) for path in image_paths]
    max_size = (
        max(frame.width for frame in frames),
        max(frame.height for frame in frames),
    )
    frames = [_pad_to_size(frame, max_size) for frame in frames]
    if column_index:
        for frame_idx, frame in enumerate(frames):
            _draw_column_labels(frame, frame_idx=frame_idx, columns=columns)
    if frame_index:
        for frame_idx, (frame, source_path) in enumerate(zip(frames, image_paths)):
            _draw_frame_label(frame, frame_idx=frame_idx, source_path=source_path)

    if output_path is None:
        output_path = image_dir.parent / "val_visualization.gif"
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration_ms = max(1, round(1000.0 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert validation visualization images into a low-FPS GIF. "
            "The input can be either visualization/val or a run directory "
            "that contains visualization/val."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing val images, or a run directory with visualization/val.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to <run>/visualization/val_visualization.gif.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second for the GIF. Default: 2.0.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern inside the image directory. Default: *.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Optionally downscale frames wider than this many pixels.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=16,
        help="Number of grid columns to stamp with the repeated frame index. Default: 16.",
    )
    parser.add_argument(
        "--no-frame-index",
        action="store_true",
        help="Do not stamp the whole-frame index onto GIF frames.",
    )
    parser.add_argument(
        "--frame-index",
        action="store_true",
        help="Also stamp the whole GIF-frame index in the top-left corner.",
    )
    parser.add_argument(
        "--no-column-index",
        action="store_true",
        help="Do not repeat the frame index above each grid column.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = make_gif(
        args.input_dir,
        args.output,
        fps=args.fps,
        pattern=args.pattern,
        max_width=args.max_width,
        frame_index=args.frame_index and not args.no_frame_index,
        column_index=not args.no_column_index,
        columns=args.cols,
    )
    print(f"Saved GIF to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
