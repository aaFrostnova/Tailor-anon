"""
Image transformations for robustness testing.

All transforms operate on numpy arrays in (C, H, W) format, float32 [0, 1].
"""

import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Tuple


def _to_pil(arr: np.ndarray) -> Image.Image:
    """(C,H,W) float32 [0,1] → PIL RGB."""
    hwc = np.clip(arr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(hwc, "RGB")


def _from_pil(img: Image.Image) -> np.ndarray:
    """PIL RGB → (C,H,W) float32 [0,1]."""
    return np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0


def jpeg_compress(arr: np.ndarray, quality: int = 80) -> np.ndarray:
    """JPEG compression at given quality level."""
    img = _to_pil(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return _from_pil(Image.open(buf).convert("RGB"))


def resize(arr: np.ndarray, scale: float = 0.5) -> np.ndarray:
    """Resize by scale factor, then resize back to original dimensions."""
    img = _to_pil(arr)
    orig_size = img.size  # (W, H)
    new_size = (max(1, int(orig_size[0] * scale)), max(1, int(orig_size[1] * scale)))
    img = img.resize(new_size, Image.BILINEAR)
    img = img.resize(orig_size, Image.BILINEAR)
    return _from_pil(img)


def center_crop(arr: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Center crop to ratio of original area, NO resize back."""
    C, H, W = arr.shape
    new_h = int(H * np.sqrt(ratio))
    new_w = int(W * np.sqrt(ratio))
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    return arr[:, top:top + new_h, left:left + new_w].copy()


def center_crop_resize(arr: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Center crop to ratio of original area, then resize back to original size."""
    C, H, W = arr.shape
    new_h = int(H * np.sqrt(ratio))
    new_w = int(W * np.sqrt(ratio))
    top = (H - new_h) // 2
    left = (W - new_w) // 2
    cropped = arr[:, top:top + new_h, left:left + new_w]
    img = _to_pil(cropped)
    img = img.resize((W, H), Image.BILINEAR)
    return _from_pil(img)


def random_crop_resize(arr: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Random position crop, then resize back to original size."""
    C, H, W = arr.shape
    new_h = int(H * np.sqrt(ratio))
    new_w = int(W * np.sqrt(ratio))
    top = np.random.randint(0, H - new_h + 1)
    left = np.random.randint(0, W - new_w + 1)
    cropped = arr[:, top:top + new_h, left:left + new_w]
    img = _to_pil(cropped)
    img = img.resize((W, H), Image.BILINEAR)
    return _from_pil(img)


def random_crop(arr: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Random position crop to ratio of original area, NO resize back."""
    C, H, W = arr.shape
    new_h = int(H * np.sqrt(ratio))
    new_w = int(W * np.sqrt(ratio))
    top = np.random.randint(0, H - new_h + 1)
    left = np.random.randint(0, W - new_w + 1)
    return arr[:, top:top + new_h, left:left + new_w].copy()


def gaussian_noise(arr: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """Add Gaussian noise with given standard deviation."""
    noise = np.random.RandomState(42).randn(*arr.shape).astype(np.float32) * sigma
    return np.clip(arr + noise, 0, 1).astype(np.float32)


def gaussian_blur(arr: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur."""
    img = _to_pil(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return _from_pil(img)


def brightness_jitter(arr: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Adjust brightness by factor."""
    img = _to_pil(arr)
    img = ImageEnhance.Brightness(img).enhance(factor)
    return _from_pil(img)


def contrast_jitter(arr: np.ndarray, factor: float = 1.2) -> np.ndarray:
    """Adjust contrast by factor."""
    img = _to_pil(arr)
    img = ImageEnhance.Contrast(img).enhance(factor)
    return _from_pil(img)


# Registry of all transforms with parameter grids
TRANSFORM_SUITE = {
    "jpeg_q95": lambda x: jpeg_compress(x, quality=95),
    "jpeg_q80": lambda x: jpeg_compress(x, quality=80),
    "jpeg_q60": lambda x: jpeg_compress(x, quality=60),
    "jpeg_q40": lambda x: jpeg_compress(x, quality=40),
    "resize_75%": lambda x: resize(x, scale=0.75),
    "resize_50%": lambda x: resize(x, scale=0.50),
    "center_crop_80%": lambda x: center_crop(x, ratio=0.80),
    "center_crop_60%": lambda x: center_crop(x, ratio=0.60),
    "center_crop_40%": lambda x: center_crop(x, ratio=0.40),
    "random_crop_80%": lambda x: random_crop(x, ratio=0.80),
    "random_crop_60%": lambda x: random_crop(x, ratio=0.60),
    "random_crop_40%": lambda x: random_crop(x, ratio=0.40),
    "center_crop_resize_80%": lambda x: center_crop_resize(x, ratio=0.80),
    "center_crop_resize_60%": lambda x: center_crop_resize(x, ratio=0.60),
    "center_crop_resize_40%": lambda x: center_crop_resize(x, ratio=0.40),
    "random_crop_resize_80%": lambda x: random_crop_resize(x, ratio=0.80),
    "random_crop_resize_60%": lambda x: random_crop_resize(x, ratio=0.60),
    "random_crop_resize_40%": lambda x: random_crop_resize(x, ratio=0.40),
    "noise_0.01": lambda x: gaussian_noise(x, sigma=0.01),
    "noise_0.05": lambda x: gaussian_noise(x, sigma=0.05),
    "noise_0.10": lambda x: gaussian_noise(x, sigma=0.10),
    "blur_0.5": lambda x: gaussian_blur(x, radius=0.5),
    "blur_1.0": lambda x: gaussian_blur(x, radius=1.0),
    "blur_2.0": lambda x: gaussian_blur(x, radius=2.0),
    "bright_+20%": lambda x: brightness_jitter(x, factor=1.2),
    "bright_-20%": lambda x: brightness_jitter(x, factor=0.8),
    "contrast_+20%": lambda x: contrast_jitter(x, factor=1.2),
    "contrast_-20%": lambda x: contrast_jitter(x, factor=0.8),
    # Compound: JPEG + crop (realistic worst case)
    "jpeg_q60+center_crop_80%": lambda x: jpeg_compress(center_crop(x, 0.8), 60),
    "jpeg_q40+random_crop_60%": lambda x: jpeg_compress(random_crop(x, 0.6), 40),
    "jpeg_q60+center_crop_resize_80%": lambda x: jpeg_compress(center_crop_resize(x, 0.8), 60),
    "jpeg_q40+random_crop_resize_60%": lambda x: jpeg_compress(random_crop_resize(x, 0.6), 40),
}
