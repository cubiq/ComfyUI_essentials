# MIT licensed code from https://github.com/li-plus/seam-carving/

from enum import Enum
from typing import Optional, Tuple

import numba as nb
import numpy as np
from scipy.ndimage import sobel

DROP_MASK_ENERGY = 1e5
KEEP_MASK_ENERGY = 1e3


class OrderMode(str, Enum):
    WIDTH_FIRST = "width-first"
    HEIGHT_FIRST = "height-first"


class EnergyMode(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


def _list_enum(enum_class) -> Tuple:
    return tuple(x.value for x in enum_class)


def _rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a grayscale image"""
    coeffs = np.array([0.2125, 0.7154, 0.0721], dtype=np.float32)
    return (rgb @ coeffs).astype(rgb.dtype)


def _get_seam_mask(src: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Convert a list of seam column indices to a mask"""
    return np.eye(src.shape[1], dtype=bool)[seam]


def _remove_seam_mask(src: np.ndarray, seam_mask: np.ndarray) -> np.ndarray:
    """Remove a seam from the source image according to the given seam_mask"""
    if src.ndim == 3:
        h, w, c = src.shape
        seam_mask = np.broadcast_to(seam_mask[:, :, None], src.shape)
        dst = src[~seam_mask].reshape((h, w - 1, c))
    else:
        h, w = src.shape
        dst = src[~seam_mask].reshape((h, w - 1))
    return dst


def _get_energy(gray: np.ndarray) -> np.ndarray:
    """Get backward energy map from the source image"""
    assert gray.ndim == 2

    gray = gray.astype(np.float32)
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy


@nb.njit(nb.int32[:](nb.float32[:, :]), cache=True)
def _get_backward_seam(energy: np.ndarray) -> np.ndarray:
    """Compute the minimum vertical seam from the backward energy map"""
    h, w = energy.shape
    inf = np.array([np.inf], dtype=np.float32)
    cost = np.concatenate((inf, energy[0], inf))
    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    for r in range(1, h):
        choices = np.vstack((cost[:-2], cost[1:-1], cost[2:]))
        min_idx = np.argmin(choices, axis=0) + base_idx
        parent[r] = min_idx
        cost[1:-1] = cost[1:-1][min_idx] + energy[r]

    c = np.argmin(cost[1:-1])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_backward_seams(
    gray: np.ndarray, num_seams: int, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute the minimum N vertical seams using backward energy"""
    h, w = gray.shape
    seams = np.zeros((h, w), dtype=bool)
    rows = np.arange(h, dtype=np.int32)
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    energy = _get_energy(gray)
    if aux_energy is not None:
        energy += aux_energy
    for _ in range(num_seams):
        seam = _get_backward_seam(energy)
        seams[rows, idx_map[rows, seam]] = True

        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

        # Only need to re-compute the energy in the bounding box of the seam
        _, cur_w = energy.shape
        lo = max(0, np.min(seam) - 1)
        hi = min(cur_w, np.max(seam) + 1)
        pad_lo = 1 if lo > 0 else 0
        pad_hi = 1 if hi < cur_w - 1 else 0
        mid_block = gray[:, lo - pad_lo : hi + pad_hi]
        _, mid_w = mid_block.shape
        mid_energy = _get_energy(mid_block)[:, pad_lo : mid_w - pad_hi]
        if aux_energy is not None:
            mid_energy += aux_energy[:, lo:hi]
        energy = np.hstack((energy[:, :lo], mid_energy, energy[:, hi + 1 :]))

    return seams


@nb.njit(
    [
        nb.int32[:](nb.float32[:, :], nb.none),
        nb.int32[:](nb.float32[:, :], nb.float32[:, :]),
    ],
    cache=True,
)
def _get_forward_seam(gray: np.ndarray, aux_energy: Optional[np.ndarray]) -> np.ndarray:
    """Compute the minimum vertical seam using forward energy"""
    h, w = gray.shape

    gray = np.hstack((gray[:, :1], gray, gray[:, -1:]))

    inf = np.array([np.inf], dtype=np.float32)
    dp = np.concatenate((inf, np.abs(gray[0, 2:] - gray[0, :-2]), inf))

    parent = np.empty((h, w), dtype=np.int32)
    base_idx = np.arange(-1, w - 1, dtype=np.int32)

    inf = np.array([np.inf], dtype=np.float32)
    for r in range(1, h):
        curr_shl = gray[r, 2:]
        curr_shr = gray[r, :-2]
        cost_mid = np.abs(curr_shl - curr_shr)
        if aux_energy is not None:
            cost_mid += aux_energy[r]

        prev_mid = gray[r - 1, 1:-1]
        cost_left = cost_mid + np.abs(prev_mid - curr_shr)
        cost_right = cost_mid + np.abs(prev_mid - curr_shl)

        dp_mid = dp[1:-1]
        dp_left = dp[:-2]
        dp_right = dp[2:]

        choices = np.vstack(
            (cost_left + dp_left, cost_mid + dp_mid, cost_right + dp_right)
        )
        min_idx = np.argmin(choices, axis=0)
        parent[r] = min_idx + base_idx
        # numba does not support specifying axis in np.min, below loop is equivalent to:
        # `dp_mid[:] = np.min(choices, axis=0)` or `dp_mid[:] = choices[min_idx, np.arange(w)]`
        for j, i in enumerate(min_idx):
            dp_mid[j] = choices[i, j]

    c = np.argmin(dp[1:-1])
    seam = np.empty(h, dtype=np.int32)
    for r in range(h - 1, -1, -1):
        seam[r] = c
        c = parent[r, c]

    return seam


def _get_forward_seams(
    gray: np.ndarray, num_seams: int, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Compute minimum N vertical seams using forward energy"""
    h, w = gray.shape
    seams = np.zeros((h, w), dtype=bool)
    rows = np.arange(h, dtype=np.int32)
    idx_map = np.broadcast_to(np.arange(w, dtype=np.int32), (h, w))
    for _ in range(num_seams):
        seam = _get_forward_seam(gray, aux_energy)
        seams[rows, idx_map[rows, seam]] = True
        seam_mask = _get_seam_mask(gray, seam)
        gray = _remove_seam_mask(gray, seam_mask)
        idx_map = _remove_seam_mask(idx_map, seam_mask)
        if aux_energy is not None:
            aux_energy = _remove_seam_mask(aux_energy, seam_mask)

    return seams


def _get_seams(
    gray: np.ndarray, num_seams: int, energy_mode: str, aux_energy: Optional[np.ndarray]
) -> np.ndarray:
    """Get the minimum N seams from the grayscale image"""
    gray = np.asarray(gray, dtype=np.float32)
    if energy_mode == EnergyMode.BACKWARD:
        return _get_backward_seams(gray, num_seams, aux_energy)
    elif energy_mode == EnergyMode.FORWARD:
        return _get_forward_seams(gray, num_seams, aux_energy)
    else:
        raise ValueError(
            f"expect energy_mode to be one of {_list_enum(EnergyMode)}, got {energy_mode}"
        )


def _reduce_width(
    src: np.ndarray,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Reduce the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if src.ndim == 2:
        gray = src
        src_h, src_w = src.shape
        dst_shape: Tuple[int, ...] = (src_h, src_w - delta_width)
    else:
        gray = _rgb2gray(src)
        src_h, src_w, src_c = src.shape
        dst_shape = (src_h, src_w - delta_width, src_c)

    to_keep = ~_get_seams(gray, delta_width, energy_mode, aux_energy)
    dst = src[to_keep].reshape(dst_shape)
    if aux_energy is not None:
        aux_energy = aux_energy[to_keep].reshape(dst_shape[:2])
    return dst, aux_energy


@nb.njit(
    nb.float32[:, :, :](nb.float32[:, :, :], nb.boolean[:, :], nb.int32), cache=True
)
def _insert_seams_kernel(
    src: np.ndarray, seams: np.ndarray, delta_width: int
) -> np.ndarray:
    """The numba kernel for inserting seams"""
    src_h, src_w, src_c = src.shape
    dst = np.empty((src_h, src_w + delta_width, src_c), dtype=src.dtype)
    for row in range(src_h):
        dst_col = 0
        for src_col in range(src_w):
            if seams[row, src_col]:
                left = src[row, max(src_col - 1, 0)]
                right = src[row, src_col]
                dst[row, dst_col] = (left + right) / 2
                dst_col += 1
            dst[row, dst_col] = src[row, src_col]
            dst_col += 1
    return dst


def _insert_seams(src: np.ndarray, seams: np.ndarray, delta_width: int) -> np.ndarray:
    """Insert multiple seams into the source image"""
    dst = src.astype(np.float32)
    if dst.ndim == 2:
        dst = dst[:, :, None]
    dst = _insert_seams_kernel(dst, seams, delta_width).astype(src.dtype)
    if src.ndim == 2:
        dst = dst.squeeze(-1)
    return dst


def _expand_width(
    src: np.ndarray,
    delta_width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Expand the width of image by delta_width pixels"""
    assert src.ndim in (2, 3) and delta_width >= 0
    if not 0 < step_ratio <= 1:
        raise ValueError(f"expect `step_ratio` to be between (0,1], got {step_ratio}")

    dst = src
    while delta_width > 0:
        max_step_size = max(1, round(step_ratio * dst.shape[1]))
        step_size = min(max_step_size, delta_width)
        gray = dst if dst.ndim == 2 else _rgb2gray(dst)
        seams = _get_seams(gray, step_size, energy_mode, aux_energy)
        dst = _insert_seams(dst, seams, step_size)
        if aux_energy is not None:
            aux_energy = _insert_seams(aux_energy, seams, step_size)
        delta_width -= step_size

    return dst, aux_energy


def _resize_width(
    src: np.ndarray,
    width: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize the width of image by removing vertical seams"""
    assert src.size > 0 and src.ndim in (2, 3)
    assert width > 0

    src_w = src.shape[1]
    if src_w < width:
        dst, aux_energy = _expand_width(
            src, width - src_w, energy_mode, aux_energy, step_ratio
        )
    else:
        dst, aux_energy = _reduce_width(src, src_w - width, energy_mode, aux_energy)
    return dst, aux_energy


def _transpose_image(src: np.ndarray) -> np.ndarray:
    """Transpose a source image in rgb or grayscale format"""
    if src.ndim == 3:
        dst = src.transpose((1, 0, 2))
    else:
        dst = src.T
    return dst


def _resize_height(
    src: np.ndarray,
    height: int,
    energy_mode: str,
    aux_energy: Optional[np.ndarray],
    step_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Resize the height of image by removing horizontal seams"""
    assert src.ndim in (2, 3) and height > 0
    if aux_energy is not None:
        aux_energy = aux_energy.T
    src = _transpose_image(src)
    src, aux_energy = _resize_width(src, height, energy_mode, aux_energy, step_ratio)
    src = _transpose_image(src)
    if aux_energy is not None:
        aux_energy = aux_energy.T
    return src, aux_energy


def _check_mask(mask: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Ensure the mask to be a 2D grayscale map of specific shape"""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expect mask to be a 2d binary map, got shape {mask.shape}")
    if mask.shape != shape:
        raise ValueError(
            f"expect the shape of mask to match the image, got {mask.shape} vs {shape}"
        )
    return mask


def _check_src(src: np.ndarray) -> np.ndarray:
    """Ensure the source to be RGB or grayscale"""
    src = np.asarray(src)
    if src.size == 0 or src.ndim not in (2, 3):
        raise ValueError(
            f"expect a 3d rgb image or a 2d grayscale image, got image in shape {src.shape}"
        )
    return src


def seam_carving(
    src: np.ndarray,
    size: Optional[Tuple[int, int]] = None,
    energy_mode: str = "backward",
    order: str = "width-first",
    keep_mask: Optional[np.ndarray] = None,
    drop_mask: Optional[np.ndarray] = None,
    step_ratio: float = 0.5,
) -> np.ndarray:
    """Resize the image using the content-aware seam-carving algorithm.

    :param src: A source image in RGB or grayscale format.
    :param size: The target size in pixels, as a 2-tuple (width, height).
    :param energy_mode: Policy to compute energy for the source image. Could be
        one of ``backward`` or ``forward``. If ``backward``, compute the energy
        as the gradient at each pixel. If ``forward``, compute the energy as the
        distances between adjacent pixels after each pixel is removed.
    :param order: The order to remove horizontal and vertical seams. Could be
        one of ``width-first`` or ``height-first``. In ``width-first`` mode, we
        remove or insert all vertical seams first, then the horizontal ones,
        while ``height-first`` is the opposite.
    :param keep_mask: An optional mask where the foreground is protected from
        seam removal. If not specified, no area will be protected.
    :param drop_mask: An optional binary object mask to remove. If given, the
        object will be removed before resizing the image to the target size.
    :param step_ratio: The maximum size expansion ratio in one seam carving step.
        The image will be expanded in multiple steps if target size is too large.
    :return: A resized copy of the source image.
    """
    src = _check_src(src)

    if order not in _list_enum(OrderMode):
        raise ValueError(
            f"expect order to be one of {_list_enum(OrderMode)}, got {order}"
        )

    aux_energy = None

    if keep_mask is not None:
        keep_mask = _check_mask(keep_mask, src.shape[:2])

        aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[keep_mask] += KEEP_MASK_ENERGY

    # remove object if `drop_mask` is given
    if drop_mask is not None:
        drop_mask = _check_mask(drop_mask, src.shape[:2])

        if aux_energy is None:
            aux_energy = np.zeros(src.shape[:2], dtype=np.float32)
        aux_energy[drop_mask] -= DROP_MASK_ENERGY

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.T

        num_seams = (aux_energy < 0).sum(1).max()
        while num_seams > 0:
            src, aux_energy = _reduce_width(src, num_seams, energy_mode, aux_energy)
            num_seams = (aux_energy < 0).sum(1).max()

        if order == OrderMode.HEIGHT_FIRST:
            src = _transpose_image(src)
            aux_energy = aux_energy.T

    # resize image if `size` is given
    if size is not None:
        width, height = size
        width = round(width)
        height = round(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"expect target size to be positive, got {size}")

        if order == OrderMode.WIDTH_FIRST:
            src, aux_energy = _resize_width(
                src, width, energy_mode, aux_energy, step_ratio
            )
            src, aux_energy = _resize_height(
                src, height, energy_mode, aux_energy, step_ratio
            )
        else:
            src, aux_energy = _resize_height(
                src, height, energy_mode, aux_energy, step_ratio
            )
            src, aux_energy = _resize_width(
                src, width, energy_mode, aux_energy, step_ratio
            )

    return src
