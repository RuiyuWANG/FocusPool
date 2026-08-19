"""Runtime-summary formatting and observation-key matching helpers."""

from __future__ import annotations

import os
import re
import textwrap
from typing import Any, Iterable, Mapping, Sequence


INNER_WIDTH = 80
FRAME_WIDTH = INNER_WIDTH + 4
KEY_WIDTH = 24


def pretty_print_nested(
    cfg: Mapping[str, Any],
    *,
    title: str | None = None,
    indent: int = 0,
    pad_before: bool = False,
    pad_after: bool = False,
) -> None:
    """Pretty-print a nested config dict in a compact runtime-summary layout."""
    lines: list[str] = []
    path_value_patterns = (
        r"[^/]+(?:/|$)",
        r"[^_]+(?:_|$)",
        r"[^-]+(?:-|$)",
    )
    enabled_states = {"enabled", "true", "active"}
    disabled_states = {"disabled", "false", "inactive"}

    class ANSI:
        RESET = "\033[0m"
        BOLD = "\033[1m"
        YELLOW = "\033[33m"
        GREEN = "\033[32m"
        RED = "\033[31m"

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def visible_len(text: str) -> int:
        return len(ansi_re.sub("", text))

    def emit(text: str = "") -> None:
        lines.append(text)

    def is_checkpoint_path(key: str, path: tuple[str, ...]) -> bool:
        return key.lower() in {"path", "ckpt_path", "checkpoint_path", "resume"} and (
            "Checkpoint" in path
        )

    def is_inline_view_block(data: Mapping[str, Any]) -> bool:
        return {"Mode", "Input Res", "Output Res"}.issubset(set(data.keys()))

    def titleize_label(value: Any) -> str:
        text = str(value)
        if "_" not in text and any(ch.isupper() for ch in text[1:]):
            return text
        return text.replace("_", " ").title()

    def normalize_status(status: Any) -> str:
        return str(status).strip().lower()

    def titleize_value(value: Any) -> str:
        return str(value).strip().replace("_", " ").title()

    def colorize(text: str, color: str, *, bold: bool = False) -> str:
        prefix = ANSI.BOLD if bold else ""
        return f"{prefix}{color}{text}{ANSI.RESET}"

    def status_parts(status: Any) -> tuple[str, str | None]:
        status_str = normalize_status(status)
        if status_str in enabled_states:
            return "✓", ANSI.GREEN
        if status_str in disabled_states:
            return "×", ANSI.RED
        return "•", None

    def status_icon(status: Any) -> str:
        mark, color = status_parts(status)
        if color is None:
            return "[•]"
        return colorize(f"[{mark}]", color)

    def status_mark(status: Any) -> str:
        mark, color = status_parts(status)
        if color is None:
            return mark
        return colorize(mark, color)

    def format_action_rep(value: Any) -> str:
        current = normalize_status(value)

        def render_mode(mode_key: str) -> str:
            mode_label = titleize_value(mode_key)
            is_selected = current == mode_key
            marker = status_icon("enabled" if is_selected else "disabled")
            if is_selected:
                return f"{marker} {colorize(mode_label, ANSI.GREEN, bold=True)}"
            return f"{marker} {mode_label}"

        return " | ".join([render_mode("absolute"), render_mode("delta")])

    def format_pooling_summary(value: Any) -> str:
        text = str(value)
        method, sep, details = text.partition(", ")
        summary = titleize_value(method)
        return f"{summary}; {details}" if sep else summary

    def format_inline_view(data: Mapping[str, Any]) -> str:
        mode = str(data.get("Mode"))
        input_res = str(data.get("Input Res"))
        output_res = str(data.get("Output Res"))
        return f"{mode}   ({input_res} -> {output_res})"

    def humanize_scalar_value(value: Any, *, key: str, path: tuple[str, ...]) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            lower = normalize_status(stripped)
            key_lower = str(key).strip().lower()

            if key_lower in {
                "action mode",
                "action rep",
                "action representation",
                "action chunk rep",
            }:
                return format_action_rep(stripped)

            if key_lower == "method" and "Pooling" in path:
                return titleize_value(stripped)

            if key_lower == "pooling" and "Encoder" in path:
                return format_pooling_summary(stripped)

            if lower in enabled_states | disabled_states:
                return f"{status_icon(lower)} {stripped.title()}"

            if key_lower in {"proprio mode", "checkpoint type"}:
                return titleize_value(stripped)

        return value

    def format_value(value: Any, *, key: str, path: tuple[str, ...]) -> Any:
        if value is None:
            if is_checkpoint_path(key, path):
                return "Randomly initialized"
            return "None"
        if isinstance(value, (list, tuple)) and len(value) == 0:
            return "None"
        if isinstance(value, float):
            return f"{value:.3f}"
        return humanize_scalar_value(value, key=key, path=path)

    def wrap_value_text(
        text: str, max_width: int, *, use_path_wrap: bool = False
    ) -> list[str]:
        if visible_len(text) <= max_width:
            return [text]

        if use_path_wrap and any(sep in text for sep in ("/", "_", "-")):
            for pattern in path_value_patterns:
                tokens = re.findall(pattern, text)
                if len(tokens) <= 1:
                    continue
                if max(len(token.rstrip()) for token in tokens) > max_width:
                    continue

                wrapped: list[str] = []
                current = ""
                for token in tokens:
                    token = token.rstrip()
                    if not current:
                        current = token
                    elif len(current) + len(token) <= max_width:
                        current += token
                    else:
                        wrapped.append(current)
                        current = token
                if current:
                    wrapped.append(current)
                if len(wrapped) > 1:
                    return wrapped

        return textwrap.wrap(
            text,
            width=max_width,
            break_long_words=True,
            break_on_hyphens=True,
        )

    def get_wrapped_value_lines(
        value: Any,
        *,
        key: str,
        path: tuple[str, ...],
        max_width: int,
    ) -> list[str]:
        formatted = format_value(value, key=key, path=path)
        text = str(formatted)
        return wrap_value_text(
            text,
            max_width,
            use_path_wrap=is_checkpoint_path(key, path) or "/" in text,
        )

    def print_kv(key: str, value: Any, ind: int, path: tuple[str, ...]) -> None:
        value_width = max(8, INNER_WIDTH - ind - KEY_WIDTH - 2)
        wrapped = get_wrapped_value_lines(
            value, key=key, path=path, max_width=value_width
        )
        key_label = titleize_label(key)
        emit(" " * ind + f"{key_label:<{KEY_WIDTH}s}: {wrapped[0]}")
        for extra in wrapped[1:]:
            emit(" " * ind + " " * (KEY_WIDTH + 2) + extra)

    def print_heading(label: str, ind: int, *, status: Any = None) -> None:
        heading_label = titleize_label(label)
        heading = (
            f"[ {heading_label} ]"
            if status is None
            else f"[{status_mark(status)} {heading_label}]"
        )
        emit(" " * ind + heading)

    def print_bullet_heading(label: str, ind: int, *, status: Any = None) -> None:
        heading_label = titleize_label(label)
        heading = (
            heading_label if status is None else f"{status_icon(status)} {heading_label}"
        )
        emit(" " * ind + heading)

    def emit_labeled_block(label: str, ind: int) -> None:
        emit(" " * ind + titleize_label(label))

    def render(data: Mapping[str, Any], ind: int, path: tuple[str, ...]) -> None:
        first = True
        for key, value in data.items():
            if isinstance(value, Mapping):
                is_top_level = len(path) == 0
                if is_top_level and not first:
                    emit()
                if is_top_level:
                    status = value.get("Status") if "Status" in value and len(value) > 1 else None
                    print_heading(str(key), ind, status=status)
                    rest = (
                        {sub_k: sub_v for sub_k, sub_v in value.items() if sub_k != "Status"}
                        if status is not None
                        else value
                    )
                    render(rest, ind, path + (str(key),))
                elif is_inline_view_block(value):
                    print_kv(str(key), format_inline_view(value), ind + 2, path)
                elif "Status" in value and len(value) > 1:
                    if not first:
                        emit()
                    print_bullet_heading(str(key), ind + 2, status=value.get("Status"))
                    rest = {sub_k: sub_v for sub_k, sub_v in value.items() if sub_k != "Status"}
                    render(rest, ind + 2, path + (str(key),))
                elif "Summary" in value and len(value) > 1:
                    print_kv(str(key), value["Summary"], ind + 2, path)
                    rest = {sub_k: sub_v for sub_k, sub_v in value.items() if sub_k != "Summary"}
                    render(rest, ind + 2, path + (str(key),))
                else:
                    emit_labeled_block(str(key), ind + 2)
                    render(value, ind + 2, path + (str(key),))
                first = False
                continue

            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                emit_labeled_block(str(key), ind + 2)
                for line in value:
                    emit(" " * (ind + 4) + line)
                first = False
                continue

            if len(path) == 0:
                if not first:
                    emit()
                print_heading(str(key), ind)
                wrapped = get_wrapped_value_lines(
                    value,
                    key=str(key),
                    path=path,
                    max_width=max(8, INNER_WIDTH - (ind + 2)),
                )
                for line in wrapped:
                    emit(" " * (ind + 2) + line)
                first = False
                continue

            print_kv(str(key), value, ind + 2, path)
            first = False

    if pad_before:
        print()

    emit()
    render(cfg, indent, path=())
    emit()

    if title is not None:
        title_text = f" {title} "
        pad = max(0, FRAME_WIDTH - 2 - len(title_text))
        top = "┌" + ("─" * (pad // 2)) + title_text + ("─" * (pad - pad // 2)) + "┐"
    else:
        top = "┌" + ("─" * (FRAME_WIDTH - 2)) + "┐"
    bottom = "└" + ("─" * (FRAME_WIDTH - 2)) + "┘"

    print(top)
    for line in lines:
        padding = max(0, INNER_WIDTH - visible_len(line))
        print("│ " + line + (" " * padding) + " │")
    print(bottom)

    if pad_after:
        print()


def fold_path_from_marker(path: str, marker: str = "experiments") -> str:
    """Fold a checkpoint path into a compact run identifier."""
    folded = os.path.normpath(str(path))
    parts = folded.split(os.sep)
    if marker in parts:
        folded = os.sep.join(parts[parts.index(marker) + 1 :])

    suffix = os.path.join("checkpoints", "latest.ckpt")
    if folded.endswith(suffix):
        folded = folded[: -len(suffix)].rstrip(os.sep)
    return folded


def is_matched_key(pattern: str, key: str, reject_pattern: str | None = None) -> bool:
    """Match an underscore-delimited token inside an observation key."""
    matched = re.search(rf"(^|_)({re.escape(pattern)})(_|$)", key) is not None
    if reject_pattern is None:
        return matched
    rejected = re.search(rf"(^|_)({re.escape(reject_pattern)})(_|$)", key) is not None
    return matched and not rejected


def find_all_keys(
    patterns: str | Sequence[str],
    keys: Iterable[str] | Mapping[str, Any],
    reject: Sequence[str] | None = None,
    strict: bool = True,
    only_one_match: bool = False,
) -> list[str] | str | None:
    """Find keys containing one of the requested observation-name tokens."""
    patterns = [patterns] if isinstance(patterns, str) else list(patterns)
    reject = [] if reject is None else list(reject)

    def contains(pattern: str, key: str) -> bool:
        return is_matched_key(pattern, key) if strict else pattern in key

    matched = [
        key
        for key in keys
        if any(contains(pattern, key) for pattern in patterns)
        and not any(contains(pattern, key) for pattern in reject)
    ]

    if only_one_match:
        if len(matched) == 0:
            return None
        if len(matched) > 1:
            raise ValueError(
                f"Multiple matches found: {matched}. Expected only one match."
            )
        return matched[0]
    return matched
