"""Defines functions that generate shapes, useful for testing purposes."""

import numpy as onp
from scipy import ndimage

PLUS_KERNEL = onp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def circle(diameter: int, padding: int) -> onp.ndarray:
  """Creates a pixelated circle with the given diameter."""
  _validate_int(diameter, padding)
  d = onp.arange(-diameter / 2 + 0.5, diameter / 2)
  distance_squared = d[:, onp.newaxis]**2 + d[onp.newaxis, :]**2
  circle = distance_squared < (diameter / 2)**2
  circle = ndimage.binary_opening(circle, PLUS_KERNEL)
  return symmetric_pad(circle, padding)


def rounded_rectangle(
    width: int,
    height: int,
    diameter: int,
    padding: int,
) -> onp.ndarray:
  """Creates a rounded rectangle."""
  _validate_int(width, height, diameter, padding)
  if width < diameter or height < diameter:
    raise ValueError(
        f"`width` and `height` must not be smaller than `diameter`, but got "
        f"{width}, {height}, and {diameter}."
    )
  c = circle(diameter, padding=0)
  pad_i = height - diameter
  pad_j = width - diameter
  rectangle = onp.zeros((height, width), dtype=bool)
  lo = diameter // 2
  i_hi = height - lo
  j_hi = width - lo
  rectangle[lo:i_hi, :] = True
  rectangle[:, lo:j_hi] = True
  rectangle |= (
      onp.pad(c, ((0, pad_i), (0, pad_j)))
      | onp.pad(c, ((pad_i, 0), (0, pad_j)))
      | onp.pad(c, ((0, pad_i), (pad_j, 0)))
      | onp.pad(c, ((pad_i, 0), (pad_j, 0)))
  )
  return symmetric_pad(rectangle, padding)


def rounded_square(width: int, diameter: int, padding: int) -> onp.ndarray:
  """Creates a single rounded square."""
  return rounded_rectangle(width, width, diameter, padding)


def checkerboard(width: int, gap: int, diameter: int) -> onp.ndarray:
  """Creates a 4x4 checkerboard, with tiles having rounded corners."""
  _validate_int(width, gap, diameter)
  square = rounded_square(width, diameter, padding=0)
  square = onp.pad(square, ((0, gap), (0, gap)))
  zeros = onp.zeros_like(square)
  x = onp.block(
      [
          [square, zeros, square, zeros], 
          [zeros, square, zeros, square], 
          [square, zeros, square, zeros], 
          [zeros, square, zeros, square], 
      ]
  )
  return onp.pad(x, ((gap, 0), (gap, 0)))


def symmetric_pad(x: onp.ndarray, padding: int) -> onp.ndarray:
  """Symmetrically pads `x` by the specified amount."""
  return onp.pad(x, ((padding, padding), (padding, padding)))


def _validate_int(*args):
  if any([not isinstance(x, int) for x in args]):
    raise ValueError(f"Expected ints but got types {[x.type for x in args]}")