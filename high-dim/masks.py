import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Literal


# ---- utilities ----
def _to_uint8_known255(mask_unknown01: np.ndarray) -> np.ndarray:
    # 1 = unknown (hole), 0 = known  -> convert to 255=known, 0=unknown
    return ((1 - mask_unknown01).astype(np.uint8)) * 255


def _scale_px(base_px: int, h: int, w: int) -> int:
    """Scale a 256-based pixel parameter to current size, min 1."""
    s = min(h, w) / 256.0
    return max(1, int(round(base_px * s)))


@dataclass
class IrregularParams:
    min_times: int
    max_times: int
    base_max_width: int  # defined at 256×256, will be scaled
    base_max_len: int  # defined at 256×256, will be scaled


@dataclass
class BoxParams:
    base_margin: int
    base_min_size: int
    base_max_size: int
    min_times: int
    max_times: int


def _random_irregular_mask(
    h: int, w: int, p: IrregularParams, rng: np.random.Generator
) -> np.ndarray:
    import cv2

    max_w = _scale_px(p.base_max_width, h, w)
    max_len = _scale_px(p.base_max_len, h, w)

    mask = np.zeros((h, w), np.float32)
    times = int(rng.integers(p.min_times, p.max_times + 1))
    for _ in range(times):
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        # 1–5 segments per stroke (LaMa-like)
        for _ in range(1 + int(rng.integers(0, 5))):
            theta = rng.random() * 2.0 * np.pi  # uniform angle in [0, 2π)
            length = int(rng.integers(max(5, max_len // 4), max_len + 1))
            brush_w = int(rng.integers(1, max_w + 1))
            x2 = int(np.clip(x + length * np.cos(theta), 0, w - 1))
            y2 = int(np.clip(y + length * np.sin(theta), 0, h - 1))
            cv2.line(mask, (x, y), (x2, y2), 1.0, brush_w)
            x, y = x2, y2
    return (mask > 0).astype(np.float32)


def _random_boxes_mask(
    h: int, w: int, p: BoxParams, rng: np.random.Generator
) -> np.ndarray:
    margin = _scale_px(p.base_margin, h, w)
    min_size = _scale_px(p.base_min_size, h, w)
    max_size = _scale_px(p.base_max_size, h, w)

    mask = np.zeros((h, w), np.float32)
    usable_w = max(0, w - 2 * margin)
    usable_h = max(0, h - 2 * margin)
    if usable_w <= 0 or usable_h <= 0:
        return mask

    # clamp max_size to fit
    max_size = min(max_size, usable_w, usable_h)
    if max_size < min_size:
        min_size = max_size
    times = int(rng.integers(p.min_times, p.max_times + 1))
    for _ in range(times):
        bw = int(rng.integers(min_size, max_size + 1)) if max_size >= 1 else 1
        bh = int(rng.integers(min_size, max_size + 1)) if max_size >= 1 else 1
        x0 = int(rng.integers(margin, margin + max(1, usable_w - bw + 1)))
        y0 = int(rng.integers(margin, margin + max(1, usable_h - bh + 1)))
        mask[y0 : y0 + bh, x0 : x0 + bw] = 1.0
    return mask


def _with_coverage_target(gen_fn, rng, min_cov: float, max_cov: float, tries: int = 10):
    """
    Try multiple seeds to keep unknown-area fraction in [min_cov, max_cov].
    Returns the closest attempt if we can't hit the window.
    """
    best = None
    best_gap = 1e9
    for _ in range(tries):
        m = gen_fn()
        cov = float(m.mean())  # since m is 0/1 unknown mask
        gap = (
            0.0
            if (min_cov <= cov <= max_cov)
            else min(abs(cov - min_cov), abs(cov - max_cov))
        )
        if gap < best_gap:
            best, best_gap = m, gap
            if gap == 0.0:
                break
    return best


# ---- fixed, size-aware generators ----
def make_wide_mask(
    h: int = 256, w: int = 256, seed: Optional[int] = None
) -> np.ndarray:
    """
    'Wide' == LaMa 'thick' scaled to any size (works at 32×32):
      base_max_width ~ 100px @256, base_max_len ~ 200px @256, few strokes + some boxes.
    Coverage targeting keeps unknown fraction ~[0.15, 0.60] (robust on tiny images).
    """
    rng = np.random.default_rng(seed)
    irr_params = IrregularParams(
        min_times=1, max_times=5, base_max_width=100, base_max_len=200
    )
    box_params = BoxParams(
        base_margin=10, base_min_size=30, base_max_size=150, min_times=1, max_times=3
    )

    def _one():
        irr = _random_irregular_mask(h, w, irr_params, rng)
        boxes = _random_boxes_mask(h, w, box_params, rng)
        return np.clip(irr + boxes, 0, 1)

    unknown = _with_coverage_target(_one, rng, min_cov=0.15, max_cov=0.60, tries=10)
    return _to_uint8_known255(unknown)


def make_narrow_mask(
    h: int = 256, w: int = 256, seed: Optional[int] = None
) -> np.ndarray:
    """
    'Narrow' == LaMa 'thin' scaled to any size (works at 32×32):
      many thin strokes: base_max_width ~ 10px @256, base_max_len ~ 40px @256, no boxes.
    Coverage targeting keeps unknown fraction ~[0.05, 0.40] (so it stays 'narrow').
    """
    rng = np.random.default_rng(seed)
    irr_params = IrregularParams(
        min_times=4, max_times=50, base_max_width=10, base_max_len=40
    )

    def _one():
        return _random_irregular_mask(h, w, irr_params, rng)

    unknown = _with_coverage_target(_one, rng, min_cov=0.05, max_cov=0.40, tries=10)
    return _to_uint8_known255(unknown)


# ---------- Structured masks from the RePaint paper ----------
def make_sr2x_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """
    Super-Resolve 2×: keep pixels on a stride-2 lattice in BOTH axes.
    Known=those lattice points; everything else unknown.
    """
    known = np.zeros((h, w), np.uint8)
    known[::2, ::2] = 255
    return known


def make_alt_lines_mask(h: int = 256, w: int = 256) -> np.ndarray:
    """
    Alternating Lines: remove every second row -> keep e.g. even rows as known.
    """
    known = np.zeros((h, w), np.uint8)
    known[::2, :] = 255
    return known


def make_expand_mask(
    h: int = 256, w: int = 256, crop: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Expand: keep a center crop 64×64 for 256×256 (scale proportionally otherwise).
    """
    if crop is None:
        # scale 64 for 256 -> 1/4 of min(h,w)
        s = max(1, (min(h, w) // 4))
        ch, cw = s, s
    else:
        ch, cw = crop
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    known = np.zeros((h, w), np.uint8)
    known[y0 : y0 + ch, x0 : x0 + cw] = 255
    return known


def make_half_mask(
    h: int = 256,
    w: int = 256,
    side: Literal["left", "right", "top", "bottom"] = "top",
) -> np.ndarray:
    """
    Half: keep the left half by default (paper & repo examples).
    """
    known = np.zeros((h, w), np.uint8)
    if side == "left":
        known[:, : w // 2] = 255
    elif side == "right":
        known[:, w // 2 :] = 255
    elif side == "top":
        known[: h // 2, :] = 255
    else:
        known[h // 2 :, :] = 255
    return known


# Convenience dispatcher
def make_mask(
    kind: Literal["wide", "narrow", "sr2x", "alt_lines", "expand", "half", "none"],
    h: int = 256,
    w: int = 256,
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    if kind == "none":
        return np.zeros((h, w), np.uint8)
    if kind == "wide":
        return make_wide_mask(h, w, seed)
    if kind == "narrow":
        return make_narrow_mask(h, w, seed)
    if kind == "sr2x":
        return make_sr2x_mask(h, w)
    if kind == "alt_lines":
        return make_alt_lines_mask(h, w)
    if kind == "expand":
        return make_expand_mask(h, w, **kwargs)
    if kind == "half":
        return make_half_mask(h, w, **kwargs)
    raise ValueError(f"Unknown kind: {kind}")


def build_mask(
    num_images: int,
    mask_type: str,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    base_seed: int = 0,
):
    m_list = []
    for i in range(num_images):
        m = make_mask(mask_type, height, width, seed=base_seed + i)
        m = (m == 255).astype(np.float32)
        m_list.append(m)

    m = np.stack(m_list, axis=0)  # [num_images, H, W]
    m = np.repeat(m[:, np.newaxis, :, :], channels, axis=1)  # [num_images, C, H, W]
    return torch.from_numpy(m).to(device)
