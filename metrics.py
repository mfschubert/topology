"""Defines functions that compute metrics for two-dimensional density arrays."""

from typing import Callable, Tuple

import cv2
import numpy as onp


# Specifies behavior in searching for the minimum length scale of an array.
DEFAULT_NON_MONOTONIC_ALLOWANCE = 5

# "Plus-shaped" kernel used througout.
_PLUS_KERNEL = onp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)

# ------------------------------------------------------------------------------
# Functions related to the length scale metric.
# ------------------------------------------------------------------------------


def minimum_length_scale(
    x: onp.ndarray,
    ignore_interfaces: bool = True,
    non_monotonic_allowance: int = DEFAULT_NON_MONOTONIC_ALLOWANCE,
) -> int:
  """Identifies the minimum length scale of features in boolean array `x`.

  The minimum length scale for an array `x` is the largest value for which there
  are no length scale violations. For a given length scale, there will be
  violations if `x` cannot be created using a "brush" with diameter equal to the
  length scale. In general, if an array can be created with a given brush, then
  its solid and void features are unchanged by binary opening operations with
  that brush.

  In some cases, an array that can be creatied with a brush of size `n` cannot
  be created with the samller brush if size `n - 1`. Further, small pixel-scale
  violations at interfaces between large features may be unimportant. Some
  allowance for these is provided via optional arguments to this function.
  
  Args:
    x: Bool-typed rank-2 array containing the features.
    ignore_interfaces: Specifies whether violations at feature interfaces are
      to be disregarded. 
    non_monotonic_allowance: See `maximum_true_arg` for details.

  Returns:
    The detected minimum length scale.
  """
  assert x.dtype == bool

  def test_fn(scale: int) -> bool:
    return ~onp.any(length_scale_violations(x, scale, ignore_interfaces))

  return maximum_true_arg(
      nearly_monotonic_fn=test_fn,
      min_arg=1,
      max_arg=max(x.shape),
      non_monotonic_allowance=non_monotonic_allowance)


def length_scale_violations(
    x: onp.ndarray,
    length_scale: int,
    ignore_interfaces: bool,
) -> onp.ndarray:
  """Identifies length scale violations of solid and void features in `x`.
  
  Optionally, the algorithm disregards pixel-scale violations that can exist at
  the boundaries of features. This gives unpredictable results when features are
  small, i.e. comparable to pixel scale.

  Args:
    x: Bool-typed rank-2 array containing the features.
    length_scale: The length scale for which violations are sought.
    ignore_interfaces: Specifies whether violations at feature interfaces are
      to be disregarded. 

  Returns:
    The array containing violations.
  """
  kernel = kernel_for_length_scale(length_scale)
  return (brush_violations(x, kernel, ignore_interfaces) |
          brush_violations(~x, kernel, ignore_interfaces))


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
  squared_distance = centers[:, onp.newaxis]**2 + centers[onp.newaxis, :]**2
  kernel = squared_distance < (length_scale / 2)**2
  # Ensure that the kernel can be realized with a width-3 brush.
  kernel = binary_opening(kernel, _PLUS_KERNEL)
  return kernel


# ------------------------------------------------------------------------------
# Array-manipulating functions backed by `cv2`.
# ------------------------------------------------------------------------------


def brush_violations(
    x: onp.ndarray,
    kernel: onp.ndarray,
    ignore_interfaces: bool,
) -> onp.ndarray:
  """Identifies brush violations of solid features in `x`.
  
  A brush violation is a pixel in `x` which is not present in the binary opening
  of `x` with structuring element `kernel`.

  Args:
    x: Bool-typed rank-2 array containing the features.
    kernel: Bool-typed rank-2 array containing the kernel.
    ignore_interfaces: Specifies whether violations at feature interfaces are
      to be disregarded. 

  Returns:
    The array containing violations.
  """
  violations = x & ~binary_opening(x, kernel)
  if ignore_interfaces:
    # If the interfaces are to be ignored, compute a mask that selects pixels
    # that are within features, i.e. they are not interface pixels. Only those
    # violations which overlap the mask are considered.
    non_interface_pixels = ~(x & ~binary_erosion(x))
    return violations & non_interface_pixels
  return violations


def binary_opening(x: onp.ndarray, kernel: onp.ndarray) -> onp.ndarray:
  """Performs binary opening with the given `kernel` and edge-mode padding.
  
  The edge-mode padding ensures that small features at the border of `x` are
  not removed.

  Args:
    x: Bool-typed rank-2 array to be transformed.
    kernel: Bool-typed rank-2 array containing the kernel.

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
  unpad_width = ((kernel.shape[0] + (kernel.shape[0] + 1) % 2,
                  kernel.shape[0] - (kernel.shape[0] + 1) % 2,),
                 (kernel.shape[1] + (kernel.shape[1] + 1) % 2,
                  kernel.shape[1] - (kernel.shape[1] + 1) % 2,),)
  opened = cv2.morphologyEx(
      src=pad_2d_edge(x, pad_width).view(onp.uint8),
      kernel=kernel.view(onp.uint8),
      op=cv2.MORPH_OPEN)
  return unpad(opened.view(bool), unpad_width)


def binary_erosion(x: onp.ndarray) -> onp.ndarray:
  """Performs binary erosion with a 2-connected kernel and edge padding."""
  assert x.dtype == bool
  pad_width = ((1, 1), (1, 1))
  eroded = cv2.erode(
      src=pad_2d_edge(x, pad_width).view(onp.uint8),
      kernel=_PLUS_KERNEL.view(onp.uint8))
  return unpad(eroded.view(bool), pad_width)


def pad_2d_edge(
    x: onp.ndarray,
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]],
) -> onp.ndarray:
  """Pads rank-2 boolean array `x` with edge pixel values."""
  assert x.dtype == bool
  ((top, bottom), (left, right)) = pad_width
  # The `copyMakeBorder` operation is equivalent to `numpy.pad`, but is faster.
  return cv2.copyMakeBorder(
      x.view(onp.uint8), 
      top=top,
      bottom=bottom,
      left=left,
      right=right,
      borderType=cv2.BORDER_REPLICATE).view(bool)


def unpad(
    x: onp.ndarray,
    pad_width: Tuple[Tuple[int, int], ...],
) -> onp.ndarray:
  """Undoes a pad operation."""
  slices = tuple([slice(pad_lo, dim - pad_hi)
                  for (pad_lo, pad_hi), dim in zip(pad_width, x.shape)])
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