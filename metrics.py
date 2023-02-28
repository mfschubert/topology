"""Defines functions that compute metrics for two-dimensional density arrays."""

from typing import Callable, Tuple

import dataclasses
import functools
import cv2
import numpy as onp


# The `IgnoreFn` accepts a length scale and an array, and returns a mask
# indicating the locations at which violations are to be ignored.
IgnoreFn = Callable[[int, onp.ndarray], onp.ndarray]

# Specifies behavior in searching for the minimum length scale of an array.
FEASIBILITY_GAP_ALLOWANCE = 5

# "Plus-shaped" kernel used througout.
PLUS_KERNEL = onp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

# "Neighbor" kernel used to identify neighbors of a pixel.
NEIGHBOR_KERNEL = onp.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], bool)

# Padding modes.
_MODE_EDGE = "edge"
_MODE_SOLID = "solid"
_MODE_VOID = "void"


# ------------------------------------------------------------------------------
# Functions related to the length scale metric.
# ------------------------------------------------------------------------------


def minimum_length_scale(
    x: onp.ndarray,
    ignore_edges: bool = True,
    feasibility_gap_allowance: int = FEASIBILITY_GAP_ALLOWANCE,
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
        ignore_edges: Determines whether violations for pixels at the edges of
            features are to be ignored.
        feasibility_gap_allowance: In checking whether a `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` used.

    Returns:
      The detected minimum length scales `(length_scale_solid, length_scale_void)`.
    """
    return (
        minimum_length_scale_solid(x, ignore_edges, feasibility_gap_allowance),
        minimum_length_scale_solid(~x, ignore_edges, feasibility_gap_allowance),
    )


def minimum_length_scale_solid(
    x: onp.ndarray,
    ignore_edges: bool,
    feasibility_gap_allowance: int,
) -> int:
    """Identifies the minimum length scale of solid features in `x`.

    Args:
        x: Bool-typed rank-2 array containing the features.
        ignore_edges: Determines whether violations for pixels at the edges of
            features are to be ignored.
        feasibility_gap_allowance: In checking whether a `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` used.

    Returns:
        The detected minimum length scale of solid features.
    """
    assert x.dtype == bool

    def test_fn(length_scale: int) -> bool:
        return ~onp.any(
            length_scale_violations_solid_with_allowance(
                x=x,
                length_scale=length_scale,
                ignore_edges=ignore_edges,
                feasibility_gap_allowance=feasibility_gap_allowance,
            )
        )

    return maximum_true_arg(
        nearly_monotonic_fn=test_fn,
        min_arg=1,
        max_arg=max(x.shape),
        non_monotonic_allowance=feasibility_gap_allowance,
    )


def length_scale_violations_solid_with_allowance(
    x: onp.ndarray,
    length_scale: int,
    ignore_edges: bool,
    feasibility_gap_allowance: int,
) -> onp.ndarray:
    """Computes the length scale violations, allowing for the feasibility gap.

    Args:
        x: Bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        ignore_edges: Determines whether violations for pixels at the edges of
            features are to be ignored.
        feasibility_gap_allowance: In checking whether a `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` used.

    Returns:
        The array containing violations.
    """
    violations = []
    for scale in range(length_scale, length_scale + feasibility_gap_allowance):
        violations.append(length_scale_violations_solid(x, scale, ignore_edges))
    violations = onp.stack(violations, axis=0)
    return onp.all(violations, axis=0)


def length_scale_violations_solid(
    x: onp.ndarray,
    length_scale: int,
    ignore_edges: bool,
) -> onp.ndarray:
    """Identifies length scale violations of solid features in `x`.

    Args:
        x: Bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        ignore_edges: Determines whether violations for pixels at the edges of
            features are to be ignored.

    Returns:
        The array containing violations.
    """
    violations = _length_scale_violations_solid(
        wrapped_x=_HashableArray(x),
        length_scale=length_scale,
        ignore_edges=ignore_edges,
    )
    assert violations.shape == x.shape
    return violations


@dataclasses.dataclass
class _HashableArray:
    """Hashable wrapper for numpy arrays."""

    array: onp.ndarray

    def __hash__(self) -> int:
        return hash((self.array.dtype, self.array.shape, self.array.tobytes()))

    def __eq__(self, other: "_HashableArray") -> bool:
        return onp.all(self.array == other.array) and (
            self.array.dtype == other.array.dtype
        )


@functools.lru_cache(maxsize=128)
def _length_scale_violations_solid(
    wrapped_x: _HashableArray,
    length_scale: int,
    ignore_edges: bool,
) -> onp.ndarray:
    """Identifies length scale violations of solid features in `x`.

    This function is strict, in the sense that no violations are ignored.

    Args:
        x: Bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        ignore_edges: Determines whether violations for pixels at the edges of
            features are to be ignored.

    Returns:
        The array containing violations.
    """
    x = wrapped_x.array
    kernel = kernel_for_length_scale(length_scale)
    violations_solid = x & ~binary_opening(x, kernel, mode=_MODE_SOLID)

    if ignore_edges:
        ignored = x & ~toplogy_preserving_binary_erosion(x)
        violations_solid = violations_solid & ~ignored

    return violations_solid


def kernel_for_length_scale(length_scale: int) -> onp.ndarray:
    """Returns an approximately circular kernel for the given `length_scale`.

    The kernel has shape `(length_scale, length_scale)`, and is `True` for pixels
    whose centers lie within the circle of radius `length_scale / 2` centered on
    the kernel. This yields a pixelated circle, which for length scales less than
    `3` will actually be square.

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
    if length_scale > 2:
        kernel = binary_opening(kernel, PLUS_KERNEL, mode=_MODE_VOID)
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


def toplogy_preserving_binary_erosion(x: onp.ndarray) -> onp.ndarray:
    """Erodes the borders of solid features in `x` while preserving the topology.

    Args:
        x: Bool-typed rank-2 array to be transformed.

    Returns:
        The array with eroded features.
    """
    eroded = cv2.erode(
        src=x.view(onp.uint8),
        kernel=PLUS_KERNEL.view(onp.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    dilated_eroded = cv2.dilate(
        src=eroded,
        kernel=PLUS_KERNEL.view(onp.uint8),
        borderType=cv2.BORDER_REPLICATE,
    )
    return (x & ~dilated_eroded.view(bool)) | eroded.view(bool)


def count_neighbors(x: onp.ndarray) -> onp.ndarray:
    """Counts the solid neighbors of each pixel in `x`."""
    assert x.dtype == bool
    return cv2.filter2D(
        src=x.view(onp.uint8),
        kernel=NEIGHBOR_KERNEL.view(onp.uint8),
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
        slice(pad_lo, dim - pad_hi) for (pad_lo, pad_hi), dim in zip(pad_width, x.shape)
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
