"""Defines functions that compute metrics for two-dimensional density arrays."""

from typing import Callable, Sequence, Tuple

import functools
import cv2
import numpy as onp


# Specifies behavior in searching for the minimum length scale of an array.
NON_MONOTONIC_ALLOWANCE = 5

# Specifies the default behavior for ignoring violations. We ignore violations
# for any solid (void) pixel having exactly two solid (void) neighbors, which
# corresponds to ignoring corners.
IGNORED_PIXEL_NEIGHBOR_COUNTS = (2,)

# "Plus-shaped" kernel used througout.
_PLUS_KERNEL = onp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

# "Neighbor" kernel used to identify neighbors of a pixel.
_NEIGHBOR_KERNEL = onp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)

# Padding modes.
_MODE_EDGE = "edge"
_MODE_SOLID = "solid"
_MODE_VOID = "void"


# ------------------------------------------------------------------------------
# Functions related to the length scale metric.
# ------------------------------------------------------------------------------


def minimum_length_scale(
    x: onp.ndarray,
    ignored_pixel_neighbor_counts: Sequence[int] = IGNORED_PIXEL_NEIGHBOR_COUNTS,
    non_monotonic_allowance: int = NON_MONOTONIC_ALLOWANCE,
) -> Tuple[int, int]:
    """Identifies the minimum length scale of solid and void features in `x`.

    The minimum length scale for solid (void) features defines the largest brush
    which can be used to recreate the solid (void) features in `x`, by convolving
    an array of "touches" with the brush kernel. In general if an array can be
    created with a given brush, then its solid and void features are unchanged by
    binary opening operations with that brush.

    In some cases, an array that can be creatied with a brush of size `n` cannot
    be created with the samller brush if size `n - 1`. Further, small pixel-scale
    violations at interfaces between large features may be unimportant. Some
    allowance for these is provided via optional arguments to this function.

    Args:
      x: Bool-typed rank-2 array containing the features.
      ignored_pixels_neighbor_counts: Specifies which pixels to ignore. Solid
            pixels with neighbor counts among the given `neighbor_counts_to_ignore`
            are marked to be ignored. See `ignored_pixels` for details.
      non_monotonic_allowance: See `maximum_true_arg` for details.

    Returns:
      The detected minimum length scales `(length_scale_solid, length_scale_void)`.
    """
    return (
        minimum_length_scale_solid(
            x, ignored_pixel_neighbor_counts, non_monotonic_allowance
        ),
        minimum_length_scale_solid(
            ~x, ignored_pixel_neighbor_counts, non_monotonic_allowance
        ),
    )


def minimum_length_scale_solid(
    x: onp.ndarray,
    ignored_pixel_neighbor_counts: Sequence[int],
    non_monotonic_allowance: int,
) -> int:
    """Identifies the minimum length scale of solid features in `x`.

    Args:
      x: Bool-typed rank-2 array containing the features.
      ignored_pixel_neighbor_counts: Specifies which pixels to ignore. Solid
            pixels with neighbor counts among the given `neighbor_counts_to_ignore`
            are marked to be ignored. See `ignored_pixels` for details.
      non_monotonic_allowance: See `maximum_true_arg` for details.

    Returns:
      The detected minimum length scale of solid features.
    """
    assert x.dtype == bool

    def test_fn(scale: int) -> bool:
        return ~onp.any(
            length_scale_violations_solid(x, scale, ignored_pixel_neighbor_counts)
        )

    return maximum_true_arg(
        nearly_monotonic_fn=test_fn,
        min_arg=1,
        max_arg=max(x.shape),
        non_monotonic_allowance=non_monotonic_allowance,
    )


def length_scale_violations_solid(
    x: onp.ndarray,
    length_scale: int,
    ignored_pixel_neighbor_counts: Sequence[int],
) -> onp.ndarray:
    """Identifies length scale violations of solid features in `x`.

    Args:
      x: Bool-typed rank-2 array containing the features.
      length_scale: The length scale for which violations are sought.
      ignored_pixel_neighbor_counts: Specifies which pixels to ignore. Solid
            pixels with neighbor counts among the given `neighbor_counts_to_ignore`
            are marked to be ignored. See `ignored_pixels` for details.

    Returns:
      The array containing violations.
    """
    ignored = ignored_pixels(x, ignored_pixel_neighbor_counts)
    kernel = kernel_for_length_scale(length_scale)
    violations_solid_padding = ~ignored & (
        x & ~binary_opening(x, kernel, mode=_MODE_SOLID)
    )
    violations_void_padding = ~ignored & (
        x & ~binary_opening(x, kernel, mode=_MODE_VOID)
    )
    return ~(~violations_solid_padding | ~violations_void_padding)


def ignored_pixels(
    x: onp.ndarray,
    ignored_pixels_neighbor_counts: Sequence[int],
) -> onp.ndarray:
    """Returns pixels for which length scale violations are to be ignored.

    The function can be configured to select pixels that are solid, with varying
    numbers of solid neighbors. Neighbors are considered in a 4-connected sense,
    i.e. a solid pixel having four solid neighbors has solid pixels above, below,
    to the left, and to the right.

    Several example configurations for ignored pixels are as follows:
        `()`: No violations are ignored.
        `(0,)`: Ignores pixels with no neighbors, i.e. isolated single pixels.
        `(1,)`: Ignores pixels with one neighbor, i.e. "single pixel penninsulas".
        `(2,)`: Ignores pixels with two neighbors, i.e. corners.
        `(0, 1, 2, 3)`: Ignores all pixels with a void neighbor, i.e. interfaces.

    Args:
        x: The array from which ignored pixels are selected.
        ignored_pixels_neighbor_counts: Specifies which pixels to ignore. Solid
            pixels with neighbor counts among the given `neighbor_counts_to_ignore`
            are marked to be ignored.

    Return:
        The ignored pixels array.
    """
    if not all([n in (0, 1, 2, 3) for n in ignored_pixels_neighbor_counts]):
        raise ValueError(
            f"Valid neighbor counts are `(0, 1, 2, 3)`, got "
            f"{ignored_pixels_neighbor_counts}."
        )

    neighbor_count = count_neighbors(x)
    return x & onp.isin(neighbor_count, ignored_pixels_neighbor_counts)


def kernel_for_length_scale(length_scale: int) -> onp.ndarray:
    """Returns an approximately circular kernel for the given `length_scale`.

    The kernel has shape `(length_scale, length_scale)`, and is `True` for pixels
    whose centers lie within the circle of radius `length_scale / 2` centered on
    the kernel. This yields a pixelated circle, which for length scales less than
    `4` will actually be square.

    Args:
      length_scale: The length scale for which the kernel is sought.

    Returns:
      The approximately circular kernel.
    """
    assert length_scale > 0
    centers = onp.arange(-length_scale / 2 + 0.5, length_scale / 2)
    squared_distance = centers[:, onp.newaxis] ** 2 + centers[onp.newaxis, :] ** 2
    kernel = squared_distance < (length_scale / 2) ** 2
    # Ensure that the kernel can be realized with a width-3 brush.
    kernel = binary_opening(kernel, _PLUS_KERNEL, mode=_MODE_VOID)
    return kernel


# ------------------------------------------------------------------------------
# Array-manipulating functions backed by `cv2`.
# ------------------------------------------------------------------------------


def binary_opening(x: onp.ndarray, kernel: onp.ndarray, mode: str) -> onp.ndarray:
    """Performs binary opening with the given `kernel` and edge-mode padding.

    The edge-mode padding ensures that small features at the border of `x` are
    not removed.

    Args:
      x: Bool-typed rank-2 array to be transformed.
      kernel: Bool-typed rank-2 array containing the kernel.
      mode: The padding mode to be used. See `pad_2d` for details.

    Returns:
      The transformed array.
    """
    assert x.ndim == 2
    assert x.dtype == bool
    assert kernel.ndim == 2
    assert kernel.dtype == bool
    pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
    # Even-size kernels lead to shifts in the image content, which we need to
    # correct by a shifted unpadding.
    unpad_width = (
        (
            kernel.shape[0] + (kernel.shape[0] + 1) % 2,
            kernel.shape[0] - (kernel.shape[0] + 1) % 2,
        ),
        (
            kernel.shape[1] + (kernel.shape[1] + 1) % 2,
            kernel.shape[1] - (kernel.shape[1] + 1) % 2,
        ),
    )
    opened = cv2.morphologyEx(
        src=pad_2d(x, pad_width, mode=mode).view(onp.uint8),
        kernel=kernel.view(onp.uint8),
        op=cv2.MORPH_OPEN,
    )
    return unpad(opened.view(bool), unpad_width)


def count_neighbors(x: onp.ndarray) -> onp.ndarray:
    """Counts the solid neighbors of each pixel in `x`."""
    assert x.dtype == bool
    return cv2.filter2D(
        src=x.view(onp.uint8),
        kernel=_NEIGHBOR_KERNEL.view(onp.uint8),
        ddepth=-1,
        borderType=cv2.BORDER_REPLICATE,
    )


def pad_2d(
    x: onp.ndarray,
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]],
    mode: str,
) -> onp.ndarray:
    """Pads rank-2 boolean array `x` with the specified mode.

    Padding may take values from the edge pixels, or be entirely solid or
    void, determined by the `mode` parameter.

    Args:
      x: The array to be padded.
      pad_width: The extent of the padding, `((i_lo, i_hi), (j_lo, j_hi))`.
      mode: Either "edge", "solid", or "void".

    Returns:
      The padded array.
    """
    assert x.dtype == bool
    ((top, bottom), (left, right)) = pad_width
    pad_fn = functools.partial(
        cv2.copyMakeBorder,
        src=x.view(onp.uint8),
        top=top,
        bottom=bottom,
        left=left,
        right=right,
    )
    if mode == _MODE_EDGE:
        return pad_fn(borderType=cv2.BORDER_REPLICATE).view(bool)
    elif mode == _MODE_SOLID:
        return pad_fn(borderType=cv2.BORDER_CONSTANT, value=1).view(bool)
    elif mode == _MODE_VOID:
        return pad_fn(borderType=cv2.BORDER_CONSTANT, value=0).view(bool)
    else:
        raise ValueError(f"Invalid `mode`, got {mode}.")


def unpad(
    x: onp.ndarray,
    pad_width: Tuple[Tuple[int, int], ...],
) -> onp.ndarray:
    """Undoes a pad operation."""
    slices = tuple(
        [
            slice(pad_lo, dim - pad_hi)
            for (pad_lo, pad_hi), dim in zip(pad_width, x.shape)
        ]
    )
    return x[slices]


# ------------------------------------------------------------------------------
# Functions that find thresholds of nearly-monotonic functions.
# ------------------------------------------------------------------------------


def maximum_true_arg(
    nearly_monotonic_fn: Callable[[onp.ndarray], bool],
    min_arg: int,
    max_arg: int,
    non_monotonic_allowance: int,
) -> int:
    """Searches for the maximum integer for which `nearly_monotonic_fn` is `True`.

    This requires `nearly_monotonic_fn` to be approximately monotonically
    decreasing, i.e. it should be `True` for small arguments and then `False` for
    large arguments. Some allowance for "noisy" behavior at the transition is
    controlled by `non_monotonic_allowance`.

    The input argument is checked in the range `[min_arg, max_arg]`, where both
    values are positive. If `test_fn` is never `True`, `min_arg` is returned.

    Note that the algorithm here assumes that `nearly_monotonic_fn` is expensive
    to evaluate with large arguments, and so a "small first" search strategy is
    employed. For this reason, `min_arg` must be positive.

    Args:
      monotonic_fn: The function for which the maximum `True` argument is sought.
      min_arg: The minimum argument. Must be positive.
      max_arg: The maximum argument. Must be greater than `min_arg.`
      non_monotonic_allowance: The number of candidate arguments where the
        function evaluates to `False` to be considered before concluding that the
        maximum `True` argument is smaller than the candidates. Must be positive.

    Returns:
      The maximum `True` argument, or `min_arg`.
    """
    assert min_arg > 0
    assert min_arg < max_arg
    assert non_monotonic_allowance > 0

    max_true_arg = min_arg - 1

    while min_arg <= max_arg:
        # We double `min_arg` rather than bisecting, as this requires fewer
        # evaluations when the minimum `True` value is close to `min_arg`.
        test_arg_start = min(min_arg * 2, (min_arg + max_arg) // 2)
        test_arg_stop = min(test_arg_start + non_monotonic_allowance, max_arg + 1)
        for test_arg in range(test_arg_start, test_arg_stop):
            result = nearly_monotonic_fn(test_arg)
            if result:
                break
        if result:
            min_arg = test_arg + 1
            max_true_arg = max(max_true_arg, test_arg)
        else:
            max_arg = test_arg_start - 1
    return max_true_arg
