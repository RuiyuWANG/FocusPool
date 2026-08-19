"""Image tensor normalization, resizing, and JPEG cache helpers."""

from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

cv2.setNumThreads(0)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ImageFormat = Literal["HWC", "CHW"]


def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Denormalize a tensor normalized with ImageNet stats."""
    if x.min().item() >= 0:
        return x
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return x * std + mean


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor with ImageNet stats."""
    if not x.is_floating_point():
        x = x.float().div(255.0)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def resize_image(image: torch.Tensor, out_res: int) -> torch.Tensor:
    """
    Resize image to out_res x out_res if needed.

    image: [N, 3, H, W]
    """
    _, _, H, W = image.shape
    if (H, W) != (out_res, out_res):
        image = F.interpolate(
            image,
            size=(out_res, out_res),
            mode="bilinear",
            align_corners=False,
        )
    return image


def decode_jpg_bytes(
    buf: bytes,
    image_size: Optional[int] = None,
    *,
    bgr_to_rgb: bool = False,
    to_float: bool = True,
    fmt: ImageFormat = "CHW",
) -> np.ndarray:
    """Decode JPEG bytes to a HWC/CHW image array."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # HWC uint8, BGR
    if img is None:
        raise ValueError("cv2.imdecode failed; buffer may be corrupted")

    if image_size is not None and (
        img.shape[0] != image_size or img.shape[1] != image_size
    ):
        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if to_float:
        img = img.astype(np.float32) / 255.0

    if fmt == "CHW":
        img = np.moveaxis(img, -1, 0)

    return img


def encode_rgb_to_jpg_bytes(img_rgb: np.ndarray, quality: int = 90) -> bytes:
    """Encode a HWC RGB image array as JPEG bytes."""
    if img_rgb.dtype != np.uint8:
        img_rgb = img_rgb.astype(np.uint8)

    ok, buf = cv2.imencode(
        ".jpg", img_rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("cv2.imencode(.jpg) failed")
    return buf.tobytes()
