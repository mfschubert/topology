"""Defines functions that generate shapes, useful for testing purposes."""

import numpy as onp
from scipy import ndimage

PLUS_KERNEL = onp.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def circle(diameter: int, padding: int) -> onp.ndarray:
    """Creates a pixelated circle with the given diameter."""
    _validate_int(diameter, padding)
    if diameter < 1:
        raise ValueError(f"`diameter` must be positive, but got {diameter}.")
    d = onp.arange(-diameter / 2 + 0.5, diameter / 2)
    distance_squared = d[:, onp.newaxis] ** 2 + d[onp.newaxis, :] ** 2
    kernel = distance_squared < (diameter / 2) ** 2
    if diameter > 2:
        # By convention we require that if the diameter is greater than `2`, 
        # the kernel must be realizable with the plus-shaped kernel.
        kernel = ndimage.binary_opening(kernel, PLUS_KERNEL)
    return symmetric_pad(kernel, padding)


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


def rounded_angled_rectangle(
    width: int,
    height: int,
    diameter: int,
    angle: float,
    padding: int,
) -> onp.ndarray:
    """Creates a rounded rectangle rotated by the specified `angle`."""
    _validate_int(width, height, diameter, padding)
    if width < diameter or height < diameter:
        raise ValueError(
            f"`width` and `height` must not be smaller than `diameter`, but got "
            f"{width}, {height}, and {diameter}."
        )
    # Generate `(i, j)` sufficient to contain the rectangle. Excess will be
    # trimmed subsequently.
    i, j = onp.meshgrid(
        onp.arange(-width - height + 0.5, width + height + 1),
        onp.arange(-width - height + 0.5, width + height + 1),
        indexing="ij",
    )
    i_rotated, j_rotated = _rotate(i, j, angle)
    rectangle = (
        (i_rotated > (-height / 2))
        & (i_rotated < (height / 2))
        & (j_rotated > (-width / 2))
        & (j_rotated < (width / 2))
    )
    # Round the corners by binary opening with a circular kernel having the
    # specified diameter.
    c = circle(diameter, padding=0)
    rectangle = ndimage.binary_opening(rectangle, c)
    rectangle = trim_zeros(rectangle)
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


def _rotate(x: onp.ndarray, y: onp.ndarray, angle: onp.ndarray) -> onp.ndarray:
    """Rotates `(x, y)` by the specified angle."""
    magnitude = onp.sqrt(x**2 + y**2)
    xy_angle = onp.angle(x + 1j * y)
    rot = magnitude * onp.exp(1j * (xy_angle + angle))
    return rot.real, rot.imag


def trim_zeros(x: onp.ndarray) -> onp.ndarray:
    """Trims the nonzero elements from `x`."""
    i, j = onp.nonzero(x)
    return x[onp.amin(i) : onp.amax(i) + 1, onp.amin(j) : onp.amax(j) + 1]


def _validate_int(*args):
    """Validates that all arguments are integers."""
    if any([not isinstance(x, int) for x in args]):
        raise ValueError(f"Expected ints but got types {[x.type for x in args]}")
