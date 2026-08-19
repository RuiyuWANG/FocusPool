"""Focused visualization helpers used by training and visual-focus eval."""

from __future__ import annotations

import os
import textwrap
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import ImageDraw
from torchvision import utils as vutils
from torchvision.transforms import functional as TF

from focuspool.util.image_ops import denorm_imagenet


def _image_batch(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or x.shape[1] != 3:
        raise ValueError(f"expected image shape [N,3,H,W], got {tuple(x.shape)}")
    x = x.float()
    if x.min().item() < 0:
        x = denorm_imagenet(x)
    return x.clamp(0, 1)


def _temporal(x: torch.Tensor, temporal_dim: int) -> torch.Tensor:
    if x.dim() >= 5:
        return x
    if x.shape[0] % temporal_dim != 0:
        raise ValueError(f"cannot reshape {tuple(x.shape)} with T={temporal_dim}")
    return x.reshape(x.shape[0] // temporal_dim, temporal_dim, *x.shape[1:])


def _draw_boxes(
    images: torch.Tensor,
    boxes: torch.Tensor,
    *,
    color: tuple[int, int, int],
    width: int = 2,
) -> torch.Tensor:
    images = images.clone()
    boxes = boxes.to(device=images.device, dtype=torch.float32)
    n, _, h, w = images.shape
    color_t = images.new_tensor(color).view(3, 1) / 255.0

    for i in range(min(n, boxes.shape[0])):
        x1, y1, x2, y2 = boxes[i].round().to(torch.int64).tolist()
        x1, x2 = sorted((max(0, min(x1, w - 1)), max(0, min(x2, w - 1))))
        y1, y2 = sorted((max(0, min(y1, h - 1)), max(0, min(y2, h - 1))))
        for k in range(max(1, int(width))):
            xl, xr = max(0, x1 - k), min(w - 1, x2 + k)
            yt, yb = max(0, y1 - k), min(h - 1, y2 + k)
            images[i, :, yt, xl : xr + 1] = color_t
            images[i, :, yb, xl : xr + 1] = color_t
            images[i, :, yt : yb + 1, xl] = color_t
            images[i, :, yt : yb + 1, xr] = color_t
    return images


def _draw_keypoints(
    images: torch.Tensor,
    keypoints: torch.Tensor,
    *,
    radius: int = 2,
) -> torch.Tensor:
    images = images.clone()
    keypoints = keypoints.to(device=images.device, dtype=torch.float32)
    n, _, h, w = images.shape
    palette = images.new_tensor(
        [
            [255, 64, 64],
            [255, 220, 64],
            [96, 255, 128],
            [224, 96, 255],
            [255, 144, 64],
            [255, 96, 192],
            [176, 255, 64],
            [255, 255, 255],
            [192, 96, 64],
        ]
    ) / 255.0
    coords = keypoints[..., :2].clone()
    mean_flags = None
    if keypoints.shape[-1] > 2:
        mean_flags = keypoints[..., 2] > 0.5
    radius_px = None
    if keypoints.shape[-1] > 3:
        radius_px = keypoints[..., 3].clone()
    if coords.numel() > 0 and coords.amin() >= -1.05 and coords.amax() <= 1.05:
        coords[..., 0] = (coords[..., 0] + 1.0) * 0.5 * (w - 1)
        coords[..., 1] = (coords[..., 1] + 1.0) * 0.5 * (h - 1)

    for i in range(min(n, coords.shape[0])):
        for j in range(coords.shape[1]):
            if not torch.isfinite(coords[i, j]).all():
                continue
            x, y = coords[i, j].round().to(torch.int64).tolist()
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            is_mean_point = mean_flags is not None and bool(mean_flags[i, j])
            if radius_px is not None and torch.isfinite(radius_px[i, j]):
                point_radius = int(round(float(radius_px[i, j].item())))
                point_radius = max(1, min(point_radius, max(h, w)))
            else:
                point_radius = int(radius)
            fill_radius = point_radius
            point_radius = point_radius + 1 if is_mean_point else point_radius
            x1 = max(0, x - point_radius)
            x2 = min(w, x + point_radius + 1)
            y1 = max(0, y - point_radius)
            y2 = min(h, y + point_radius + 1)
            if is_mean_point:
                border = images.new_tensor([0.0, 0.0, 0.0]).view(3, 1, 1)
                fill = images.new_tensor([1.0, 1.0, 1.0]).view(3, 1, 1)
                images[i, :, y1:y2, x1:x2] = border
                ix1 = max(0, x - fill_radius)
                ix2 = min(w, x + fill_radius + 1)
                iy1 = max(0, y - fill_radius)
                iy2 = min(h, y + fill_radius + 1)
                images[i, :, iy1:iy2, ix1:ix2] = fill
            else:
                color = palette[j % palette.shape[0]].view(3, 1, 1)
                images[i, :, y1:y2, x1:x2] = color
    return images


def _overlay_mask(
    images: torch.Tensor,
    mask: torch.Tensor,
    *,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),
    alpha: float = 0.5,
    blackout: bool = False,
    mode: str = "nearest",
) -> torch.Tensor:
    mask = mask.to(device=images.device, dtype=images.dtype)
    if mask.shape[-2:] != images.shape[-2:]:
        mask = F.interpolate(mask, size=images.shape[-2:], mode=mode)
    mask = mask.clamp(0, 1)
    vmax = mask.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    mask = mask / vmax

    if blackout:
        return images * mask + 0.25 * images.new_ones(images.shape) * (1 - mask)

    color_t = images.new_tensor(color).view(1, 3, 1, 1)
    return (images * (1 - alpha * mask) + color_t * (alpha * mask)).clamp(0, 1)


def _pad_even(frame_hwc: torch.Tensor) -> torch.Tensor:
    pad_h = frame_hwc.shape[0] % 2
    pad_w = frame_hwc.shape[1] % 2
    if not pad_h and not pad_w:
        return frame_hwc
    return F.pad(frame_hwc, (0, 0, 0, pad_w, 0, pad_h), value=0)


def plot_switch_points(
    scores: np.ndarray | torch.Tensor,
    switch_idx: Sequence[int],
    images: Optional[np.ndarray | torch.Tensor] = None,
    *,
    episode_index: Optional[int] = None,
    include_start: bool = True,
    title: Optional[str] = None,
    ylabel: str = "Score",
):
    scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)
    idx = np.array(sorted({int(i) for i in switch_idx}), dtype=np.int64)
    idx = idx[(0 <= idx) & (idx < scores_np.shape[0])]
    if include_start and (idx.size == 0 or idx[0] != 0):
        idx = np.concatenate([np.array([0], dtype=np.int64), idx])

    plot_title = title or "Context-change score"
    if episode_index is not None:
        plot_title = f"{plot_title} (episode {episode_index})"

    if images is None:
        fig, ax = plt.subplots(figsize=(9, 3))
        axes = [ax]
    else:
        fig = plt.figure(figsize=(4 * max(len(idx), 3), 6))
        grid = fig.add_gridspec(2, max(len(idx), 1), height_ratios=[2, 3])
        axes = [fig.add_subplot(grid[0, :])]

    axes[0].plot(scores_np, linewidth=1.5)
    axes[0].set_title(plot_title)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel(ylabel)
    for i in idx:
        axes[0].axvline(int(i), linestyle="--", linewidth=1)
        axes[0].scatter([int(i)], [scores_np[int(i)]], zorder=3)

    if images is not None:
        if isinstance(images, torch.Tensor):
            frames = images.detach().cpu()
        else:
            frames = torch.as_tensor(images)
        if frames.dim() == 4 and frames.shape[1] == 3:
            frames = _image_batch(frames).permute(0, 2, 3, 1)
        for col, frame_idx in enumerate(idx):
            ax = fig.add_subplot(grid[1, col])
            ax.imshow(frames[int(frame_idx)].numpy())
            ax.set_title(f"t={int(frame_idx)}")
            ax.axis("off")
    return fig, idx


def visualize(
    views,
    temporal_dim: int = 1,
    save_dir: Optional[str] = None,
    step: Optional[int] = None,
    num_viz: int = 16,
    padding: int = 2,
    text: Optional[str | Sequence[str]] = None,
    mask_alpha: float = 0.5,
    mask_color: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    rows = []
    box_colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 220, 0)]

    for image, mask, boxes, keypoints in views:
        image = _temporal(image, temporal_dim)
        mask = _temporal(mask, temporal_dim) if mask is not None else None
        keypoints = (
            _temporal(keypoints, temporal_dim) if keypoints is not None else None
        )
        box_list = []
        if boxes is not None:
            box_list = boxes if isinstance(boxes, list) else [boxes]
            box_list = [_temporal(box, temporal_dim) for box in box_list]

        batch = min(image.shape[0], int(num_viz))
        image = image[:batch]
        if mask is not None:
            mask = mask[:batch]
        if keypoints is not None:
            keypoints = keypoints[:batch]
        box_list = [box[:batch] for box in box_list]

        for t in range(image.shape[1]):
            row = _image_batch(image[:, t])
            for i, box in enumerate(box_list):
                row = _draw_boxes(
                    row,
                    box[:, t],
                    color=box_colors[i % len(box_colors)],
                )
            if mask is not None:
                row = _overlay_mask(
                    row,
                    mask[:, t],
                    color=mask_color,
                    alpha=mask_alpha,
                )
            if keypoints is not None:
                row = _draw_keypoints(row, keypoints[:, t])
            if text is not None:
                if isinstance(text, str):
                    labels = [text] * batch
                else:
                    labels = list(text)[:batch]
                pil_frames = []
                for frame, label in zip(row.detach().cpu(), labels):
                    pil = TF.to_pil_image(frame)
                    draw = ImageDraw.Draw(pil)
                    draw.multiline_text(
                        (5, 5),
                        textwrap.fill(str(label), width=48),
                        fill=(255, 255, 255),
                    )
                    pil_frames.append(TF.to_tensor(pil))
                row = torch.stack(pil_frames, dim=0).to(row.device)
            rows.append(row)

    grid = vutils.make_grid(
        torch.cat(rows, dim=0),
        nrow=rows[0].shape[0],
        padding=padding,
    )
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = "viz.png" if step is None else f"train_step_{int(step)}_viz.png"
        vutils.save_image(grid, os.path.join(save_dir, name))
    return grid


def _smooth_mask(mask: torch.Tensor, momentum: float) -> torch.Tensor:
    if momentum is None or momentum <= 0:
        return mask
    out = mask.clone()
    prev = out[0]
    for t in range(1, out.shape[0]):
        prev = momentum * prev + (1 - momentum) * out[t]
        out[t] = prev
    return out.clamp(0, 1)


def smooth_box(box: torch.Tensor, s: float) -> torch.Tensor:
    if s is None or s <= 0:
        return box
    x = box.float()
    out = x.clone()
    prev = x[0]
    for t in range(1, x.shape[0]):
        prev = float(s) * prev + (1 - float(s)) * x[t]
        out[t] = prev
    return out


def visualize_trajectory(
    images: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    boxes: Optional[list[torch.Tensor]] = None,
    reference_boxes: Optional[list[torch.Tensor]] = None,
    reference_masks: Optional[list[torch.Tensor]] = None,
    comparison_boxes: Optional[list[torch.Tensor]] = None,
    eih_images: Optional[torch.Tensor] = None,
    eih_mask: Optional[torch.Tensor] = None,
    eih_boxes: Optional[list[torch.Tensor]] = None,
    gripper_opening: Optional[torch.Tensor] = None,
    save_dir: Optional[str] = None,
    step: Optional[int] = None,
    prefix: str = "frame",
    text: Optional[str | Sequence[str]] = None,
    draw_gripper_bar: bool = True,
    box_smoothing: Optional[float] = None,
    mask_smoothing: Optional[float] = None,
    box_padding: int = 8,
    mask_interp: str = "nearest",
    blackout: bool = False,
    save_video: bool = False,
):
    images = _image_batch(images)
    eih_images = _image_batch(eih_images) if eih_images is not None else None
    frames = []
    total = images.shape[0]

    if mask is not None:
        mask = _smooth_mask(mask.float(), mask_smoothing)
    if eih_mask is not None:
        eih_mask = _smooth_mask(eih_mask.float(), mask_smoothing)
    if reference_masks is not None:
        reference_masks = [
            _smooth_mask(m.float(), mask_smoothing) for m in reference_masks
        ]

    pad = images.new_tensor([-box_padding, -box_padding, box_padding, box_padding])
    boxes = [
        smooth_box(box.to(torch.float32) + pad, box_smoothing)
        for box in boxes or []
    ]
    reference_boxes = [box.to(torch.float32) for box in reference_boxes or []]
    comparison_boxes = [box.to(torch.float32) for box in comparison_boxes or []]
    eih_boxes = [
        smooth_box(box.to(torch.float32) + pad, box_smoothing)
        for box in eih_boxes or []
    ]
    labels = [text] * total if isinstance(text, str) else list(text or [])
    if gripper_opening is not None and gripper_opening.dim() == 2:
        gripper_opening = gripper_opening[:, 0]

    if save_video and save_dir is None:
        raise ValueError("save_dir is required when save_video=True")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    pad_col = images.new_zeros((1, 3, images.shape[-2], 16))

    for t in range(total):
        left = images[t : t + 1].clone()
        for box in boxes:
            left = _draw_boxes(left, box[t : t + 1], color=(255, 0, 0))
        for box in reference_boxes:
            left = _draw_boxes(left, box[t : t + 1], color=(0, 255, 0))
        for box in comparison_boxes:
            left = _draw_boxes(left, box[t : t + 1], color=(255, 220, 0), width=3)
        for ref_mask in reference_masks or []:
            left = _overlay_mask(
                left,
                ref_mask[t : t + 1],
                color=(0.0, 1.0, 0.0),
                alpha=0.4,
                mode="nearest",
            )

        if labels or (draw_gripper_bar and gripper_opening is not None):
            pil = TF.to_pil_image(left[0].cpu())
            draw = ImageDraw.Draw(pil)
            if labels:
                draw.multiline_text(
                    (5, 5),
                    textwrap.fill(str(labels[t]), width=60),
                    fill=(255, 255, 255),
                )
            if draw_gripper_bar and gripper_opening is not None:
                value = float(gripper_opening[t].clamp(-1, 1))
                pct = int(round((value + 1.0) * 50.0))
                x0, y0 = 6, pil.height - 22
                width, height = int(pil.width * 0.3), 10
                draw.text((x0, y0 - 12), f"Grip {pct}%", fill=(180, 180, 180))
                draw.rectangle(
                    [x0, y0, x0 + width, y0 + height],
                    outline=(180, 180, 180),
                )
                draw.rectangle(
                    [x0, y0, x0 + int(width * pct / 100.0), y0 + height],
                    fill=(180, 180, 180),
                )
            left = TF.to_tensor(pil).to(images.device).unsqueeze(0)

        right = images[t : t + 1].clone()
        if mask is not None:
            right = _overlay_mask(
                right,
                mask[t : t + 1],
                blackout=blackout,
                mode=mask_interp,
            )
        frame = torch.cat([left, pad_col, right], dim=-1)

        if eih_images is not None:
            eih = eih_images[t : t + 1]
            eih_left = eih.clone()
            for box in eih_boxes:
                eih_left = _draw_boxes(eih_left, box[t : t + 1], color=(255, 0, 0))
            eih_right = eih.clone()
            if eih_mask is not None:
                eih_right = _overlay_mask(
                    eih_right,
                    eih_mask[t : t + 1],
                    blackout=blackout,
                    mode=mask_interp,
                )
            eih_pad_col = eih.new_zeros((1, 3, eih.shape[-2], 16))
            eih_row = torch.cat([eih_left, eih_pad_col, eih_right], dim=-1)
            spacer = frame.new_zeros((1, 3, 12, frame.shape[-1]))
            frame = torch.cat([frame, spacer, eih_row], dim=-2)

        if save_video:
            u8 = (frame[0].clamp(0, 1) * 255).round().to(torch.uint8)
            frames.append(_pad_even(u8.permute(1, 2, 0).cpu()))
        elif save_dir is not None:
            suffix = f"step{step}_" if step is not None else ""
            vutils.save_image(
                frame,
                os.path.join(save_dir, f"{prefix}_{suffix}t{t:04d}.png"),
            )

    if save_video:
        name = prefix if step is None else f"{prefix}_step{step}"
        video_path = os.path.join(save_dir, f"{name}.mp4")
        torchvision.io.write_video(video_path, torch.stack(frames, dim=0), fps=30)


@torch.no_grad()
def save_attention_heads_video(
    images: torch.Tensor,
    attn: torch.Tensor,
    head_score: torch.Tensor,
    save_path: str,
    *,
    grid_hw: int = 14,
    fps: int = 30,
    alpha: float = 0.55,
    tile_scale: float = 0.45,
    save_frames_every=None,
):
    images = _image_batch(images)
    attn = attn[:, :, 0].float()
    attn = attn.reshape(attn.shape[0], attn.shape[1], grid_hw, grid_hw)
    attn = F.interpolate(
        attn.reshape(-1, 1, grid_hw, grid_hw),
        size=images.shape[-2:],
        mode="nearest",
    ).reshape(attn.shape[0], attn.shape[1], 1, *images.shape[-2:])
    weights = head_score[:, :, 0, 0].float().clamp_min(0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    frame_dir = os.path.splitext(save_path)[0]
    dump_frames = save_frames_every is not None and int(save_frames_every) > 0
    if dump_frames:
        os.makedirs(frame_dir, exist_ok=True)

    frames = []
    for t in range(images.shape[0]):
        tiles = []
        for h in range(attn.shape[1]):
            tile = _overlay_mask(
                images[t : t + 1],
                attn[t, h : h + 1],
                alpha=alpha,
                color=(1.0, 0.0, 0.0),
            )
            pil = TF.to_pil_image(tile[0].cpu())
            ImageDraw.Draw(pil).text(
                (5, 5),
                f"h{h} {weights[t, h].item():.2f}",
                fill=(255, 255, 255),
            )
            tiles.append(TF.to_tensor(pil))
        grid = vutils.make_grid(torch.stack(tiles), nrow=attn.shape[1], padding=4)
        if tile_scale != 1.0:
            size = [
                max(8, int(grid.shape[-2] * tile_scale)),
                max(8, int(grid.shape[-1] * tile_scale)),
            ]
            grid = F.interpolate(grid[None], size=size, mode="nearest")[0]
        frame = (grid.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        frame = _pad_even(frame.cpu())
        frames.append(frame)
        if dump_frames and t % int(save_frames_every) == 0:
            torchvision.io.write_png(
                frame.permute(2, 0, 1),
                os.path.join(frame_dir, f"{t:06d}.png"),
            )
    torchvision.io.write_video(save_path, torch.stack(frames, dim=0), fps=fps)
