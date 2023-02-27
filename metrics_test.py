"""Tests for `metrics`."""

import itertools
import numpy as onp
import parameterized
from scipy import ndimage
import unittest

import metrics

TEST_ARRAY_4_5 = onp.array(
    [  # Solid features feasible with circle-4, void with circle-5.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_5 = onp.array(
    [  # Solid features feasible with circle-5, void with circle-5.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_3 = onp.array(
    [  # Solid features feasible with circle-5, void with circle-3.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_WITH_DEFECT = onp.array(
    [  # Mostly feasible with diameter-4 brush.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # One pixel here is defective.
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAYS = [TEST_ARRAY_4_5, TEST_ARRAY_5_5, TEST_ARRAY_5_3, TEST_ARRAY_5_WITH_DEFECT]


TEST_KERNEL_4 = onp.array(
    [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype="bool"
)
TEST_KERNEL_5 = onp.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype="bool",
)
TEST_KERNEL_4_3_ASYMMETRIC = onp.array(
    [[0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype="bool"
)
TEST_KERNELS = [TEST_KERNEL_4, TEST_KERNEL_5, TEST_KERNEL_4_3_ASYMMETRIC]


IGNORE_NONE = ()
IGNORE_PENINSULAS = (1,)
IGNORE_INTERFACES = (0, 1, 2, 3)
IGNORE_CORNERS = (2,)


class LengthScaleTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (TEST_ARRAY_4_5, 4, 5),
            (TEST_ARRAY_5_5, 5, 5),
            (TEST_ARRAY_5_3, 5, 3),
        ]
    )
    def test_length_scale_matches_expected(self, x, expected_solid, expected_void):
        length_scale_solid, length_scale_void = metrics.minimum_length_scale(
            x, IGNORE_NONE
        )
        self.assertEqual(length_scale_solid, expected_solid)
        self.assertEqual(length_scale_void, expected_void)

    @parameterized.parameterized.expand([[i] for i in range(5, 20)])
    def test_circle_has_expected_length_scale(self, length_scale):
        # Make an array that has a single solid circular feature with diameter equal
        # to the length scale. Pad to make sure the feature is isolated.
        x = metrics.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode="constant")
        length_scale_solid, length_scale_void = metrics.minimum_length_scale(
            x, IGNORE_NONE
        )
        self.assertEqual(length_scale_solid, length_scale)
        self.assertEqual(length_scale_void, min(x.shape))

    @parameterized.parameterized.expand([[i] for i in range(5, 20)])
    def test_hole_has_expected_length_scale(self, length_scale):
        # Make an array that has a single void circular feature with diameter equal
        # to the length scale. Pad to make sure the feature is isolated.
        x = metrics.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode="constant")
        length_scale_solid, length_scale_void = metrics.minimum_length_scale(
            ~x, IGNORE_NONE
        )
        self.assertEqual(length_scale_void, length_scale)
        self.assertEqual(length_scale_solid, min(x.shape))

    def test_brush_violations_with_interface_defects(self):
        # Assert that there are violations in the defective array.
        assert onp.any(
            metrics.length_scale_violations_solid(
                TEST_ARRAY_5_WITH_DEFECT, 4, IGNORE_NONE
            )
        )
        # Assert that there are no violations if we ignore interfaces.
        assert not onp.any(
            metrics.length_scale_violations_solid(
                TEST_ARRAY_5_WITH_DEFECT, 4, IGNORE_INTERFACES
            )
        )
        # Assert that there are violations if we only ignore corners.
        assert onp.any(
            metrics.length_scale_violations_solid(
                TEST_ARRAY_5_WITH_DEFECT, 4, IGNORE_CORNERS
            )
        )

    def test_solid_feature_shallow_incidence(self):
        # Checks that the length scale for a design having a solid feature that
        # is incident on the design edge with a very shallow angle has a length
        # scale equal to the size of the design.
        x = onp.ones((70, 70), dtype=bool)
        x[-1, 10:] = 0
        x[-2, 20:] = 0
        length_scale_solid, length_scale_void = metrics.minimum_length_scale(x)
        self.assertEqual(length_scale_solid, 70)
        self.assertEqual(length_scale_void, 70)


class KernelTest(unittest.TestCase):
    @parameterized.parameterized.expand([(4, TEST_KERNEL_4), (5, TEST_KERNEL_5)])
    def test_kernel_matches_expected(self, length_scale, expected):
        onp.testing.assert_array_equal(
            metrics.kernel_for_length_scale(length_scale), expected
        )


class IgnorePixelsTest(unittest.TestCase):
    def test_ignore_none(self):
        ignored = metrics.ignored_pixels(TEST_ARRAY_5_WITH_DEFECT, ())
        onp.testing.assert_array_equal(ignored, onp.zeros_like(ignored))

    def test_ignore_peninsula(self):
        ignored = metrics.ignored_pixels(TEST_ARRAY_5_WITH_DEFECT, IGNORE_PENINSULAS)
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(ignored, expected)

    def test_ignore_corner(self):
        ignored = metrics.ignored_pixels(TEST_ARRAY_5_WITH_DEFECT, IGNORE_CORNERS)
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(ignored, expected)

    def test_ignore_interfaces(self):
        ignored = metrics.ignored_pixels(TEST_ARRAY_5_WITH_DEFECT, IGNORE_INTERFACES)
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(ignored, expected)


# ------------------------------------------------------------------------------
# Tests for array-manipulating functions.
# ------------------------------------------------------------------------------


class MorphologyOperationsTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        list(itertools.product(TEST_KERNELS, TEST_ARRAYS))
    )
    def test_opening_matches_scipy(self, x, kernel):
        pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
        x_padded = onp.pad(x, pad_width, mode="edge")
        expected = ndimage.binary_opening(x_padded, kernel)
        expected = expected[
            pad_width[0][0] : expected.shape[0] - pad_width[0][1],
            pad_width[1][0] : expected.shape[1] - pad_width[1][1],
        ]
        actual = metrics.binary_opening(x, kernel, mode="edge")
        onp.testing.assert_array_equal(expected, actual)

    def test_opening_removes_small_features(self):
        actual = metrics.binary_opening(TEST_ARRAY_4_5, TEST_KERNEL_5, mode="edge")
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(expected, actual)

    def test_count_neighbors(self):
        neighbors = metrics.count_neighbors(TEST_ARRAY_5_WITH_DEFECT)
        expected = onp.array(
            [  
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 2, 3, 2, 2, 0, 0, 0, 0, 0, 0, 0],
                [1, 2, 4, 4, 4, 1, 1, 0, 0, 0, 0, 1, 1],
                [1, 3, 4, 4, 3, 3, 0, 0, 0, 0, 2, 2, 3],
                [1, 2, 4, 4, 4, 1, 1, 0, 0, 1, 2, 4, 4],
                [0, 2, 2, 3, 2, 2, 0, 0, 0, 1, 3, 4, 4],
                [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 3, 4, 4],
            ],
        )
        onp.testing.assert_array_equal(neighbors, expected)

    @parameterized.parameterized.expand(
        [
            [((0, 0), (0, 0))],
            [((1, 5), (2, 4))],
            [((4, 2), (5, 1))],
        ],
    )
    def test_pad_2d_matches_numpy(self, pad_width):
        onp.random.seed(0)
        x = onp.random.rand(20, 30) > 0.5  # Random binary array.
        with self.subTest("edge"):
            expected = onp.pad(x, pad_width, mode="edge")
            actual = metrics.pad_2d(x, pad_width, mode="edge")
            onp.testing.assert_array_equal(expected, actual)
        with self.subTest("solid"):
            expected = onp.pad(x, pad_width, constant_values=True)
            actual = metrics.pad_2d(x, pad_width, mode="solid")
            onp.testing.assert_array_equal(expected, actual)
        with self.subTest("void"):
            expected = onp.pad(x, pad_width, constant_values=False)
            actual = metrics.pad_2d(x, pad_width, mode="void")
            onp.testing.assert_array_equal(expected, actual)

    @parameterized.parameterized.expand(
        [
            [((0, 0), (0, 0))],
            [((1, 5), (2, 4))],
            [((4, 2), (5, 1))],
        ],
    )
    def test_unpad(self, pad_width):
        x = onp.arange(200).reshape(10, 20)
        expected = x[
            pad_width[0][0] : x.shape[0] - pad_width[0][1],
            pad_width[1][0] : x.shape[1] - pad_width[1][1],
        ]
        actual = metrics.unpad(x, pad_width)
        onp.testing.assert_equal(expected.shape, actual.shape)
        onp.testing.assert_array_equal(expected, actual)


# ------------------------------------------------------------------------------
# Tests for `maximum_true_arg`.
# ------------------------------------------------------------------------------


class MaximumTrueArgTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            # 1  2  3  4  5  6  7  8  9  10 11
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 1, 11, 1, 10),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 1, 3),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 11, 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 1, 8, 1, 0),  # No `True` values.
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1, 8, 1, 8),  # All `True` values.
        ]
    )
    def test_finds_maximum_true_arg(
        self,
        sequence,
        min_arg,
        max_arg,
        allowance,
        expected,
    ):
        fn = lambda i: bool(sequence[i - 1])
        actual = metrics.maximum_true_arg(fn, min_arg, max_arg, allowance)
        assert expected == actual
