"""Tests for `metrics`."""

import itertools
import numpy as onp
import parameterized
from scipy import ndimage
import unittest

import metrics

TEST_ARRAY_4 = onp.array([  # Feasible with a diameter-4 brush.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
], dtype=bool)
TEST_ARRAY_5 = onp.array([  # Feasible with a diameter-5 brush.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
], dtype=bool)
TEST_ARRAY_5_WITH_DEFECT = onp.array([  # Mostly feasible with diameter-4 brush.
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # One pixel here is defective.
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
], dtype=bool)
TEST_ARRAYS = [TEST_ARRAY_4, TEST_ARRAY_5, TEST_ARRAY_5_WITH_DEFECT]

TEST_KERNEL_4 = onp.array(
    [[0, 1, 1, 0],
     [1, 1, 1, 1],
     [1, 1, 1, 1],
     [0, 1, 1, 0]], dtype='bool')
TEST_KERNEL_5 = onp.array(
    [[0, 1, 1, 1, 0],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1],
     [0, 1, 1, 1, 0]], dtype='bool')
TEST_KERNEL_4_3_ASYMMETRIC = onp.array(
    [[0, 1, 0],
     [1, 1, 1],
     [1, 1, 1],
     [0, 1, 0]], dtype='bool')
TEST_KERNELS = [TEST_KERNEL_4, TEST_KERNEL_5, TEST_KERNEL_4_3_ASYMMETRIC]


class LengthScaleTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        [
            (TEST_ARRAY_4, 4),
            (TEST_ARRAY_5, 5),
        ]
    )
    def test_length_scale_matches_expected(self, x, expected):
        assert expected == metrics.minimum_length_scale(x, ignore_interfaces=False)

    @parameterized.parameterized.expand([[i] for i in range(5, 20)])
    def test_circle_has_expected_length_scale(self, length_scale):
        # Make an array that has a single solid circular feature with diameter equal
        # to the length scale. Pad to make sure the feature is isolated.
        x = metrics.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode='constant')
        assert length_scale == metrics.minimum_length_scale(x)

    @parameterized.parameterized.expand([[i] for i in range(5, 20)])
    def test_hole_has_expected_length_scale(self, length_scale):
        # Make an array that has a single void circular feature with diameter equal
        # to the length scale. Pad to make sure the feature is isolated.
        x = metrics.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode='constant')
        x = ~x
        assert length_scale == metrics.minimum_length_scale(x)

    def test_brush_violations_with_interface_defects(self):
        # Assert that there are violations in the defective array.
        assert onp.any(metrics.brush_violations(
            TEST_ARRAY_5_WITH_DEFECT, TEST_KERNEL_4, ignore_interfaces=False))
        # Assert that there are no violations if we ignore interfaces.
        assert not onp.any(metrics.brush_violations(
            TEST_ARRAY_5_WITH_DEFECT, TEST_KERNEL_4, ignore_interfaces=True))

    @parameterized.parameterized.expand(
        [
            (TEST_ARRAY_4, TEST_KERNEL_4),
            (TEST_ARRAY_5, TEST_KERNEL_5),
        ]
    )
    def test_no_brush_violations_with_feasible_arrays(self, x, kernel):
        onp.testing.assert_array_equal(
            onp.zeros_like(x),
            metrics.brush_violations(x, kernel, ignore_interfaces=False))


class KernelTest(unittest.TestCase):
    @parameterized.parameterized.expand([(4, TEST_KERNEL_4), (5, TEST_KERNEL_5)])
    def test_kernel_matches_expected(self, length_scale, expected):
        onp.testing.assert_array_equal(
            metrics.kernel_for_length_scale(length_scale), expected)


# ------------------------------------------------------------------------------
# Tests for array-manipulating functions.
# ------------------------------------------------------------------------------


class MorphologyOperationsTest(unittest.TestCase):
    @parameterized.parameterized.expand([[arr] for arr in TEST_ARRAYS])
    def test_erosion_matches_scipy(self, x):
        x_padded = onp.pad(x, ((1, 1), (1, 1)), mode='edge')
        kernel = onp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
        expected = ndimage.binary_erosion(x_padded, kernel)
        expected = expected[1:-1, 1:-1]
        actual = metrics.binary_erosion(x)
        onp.testing.assert_array_equal(expected, actual)


    @parameterized.parameterized.expand(list(itertools.product(TEST_KERNELS, TEST_ARRAYS)))
    def test_opening_matches_scipy(self, x, kernel):
        pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
        x_padded = onp.pad(x, pad_width, mode='edge')
        expected = ndimage.binary_opening(x_padded, kernel)
        expected = expected[pad_width[0][0]:expected.shape[0] - pad_width[0][1],
                            pad_width[1][0]:expected.shape[1] - pad_width[1][1]]
        actual = metrics.binary_opening(x, kernel)
        onp.testing.assert_array_equal(expected, actual)

    def test_opening_removes_small_features(self):
        actual = metrics.binary_opening(TEST_ARRAY_4, TEST_KERNEL_5)
        expected = onp.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ], dtype=bool)
        onp.testing.assert_array_equal(expected, actual)


    @parameterized.parameterized.expand(
        [
            [((0, 0), (0, 0))],
            [((1, 5), (2, 4))],
            [((4, 2), (5, 1))],
        ],
    )
    def test_pad_2d_edge_matches_numpy(self, pad_width):
        onp.random.seed(0)
        x = onp.random.rand(20, 30) > 0.5  # Random binary array.
        expected = onp.pad(x, pad_width, mode='edge')
        actual = metrics.pad_2d_edge(x, pad_width)
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
        expected = x[pad_width[0][0]:x.shape[0] - pad_width[0][1],
                    pad_width[1][0]:x.shape[1] - pad_width[1][1]]
        actual = metrics.unpad(x, pad_width)
        onp.testing.assert_equal(expected.shape, actual.shape)
        onp.testing.assert_array_equal(expected, actual)


# ------------------------------------------------------------------------------
# Tests for `maximum_true_arg`.
# ------------------------------------------------------------------------------


class MaximumTrueArgTest(unittest.TestCase):
    @parameterized.parameterized.expand(
        #  1  2  3  4  5  6  7  8  9  10 11
        [
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 1, 11, 1, 10),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 1, 3),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 11, 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 1, 8, 1, 0),  # No `True` values.
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1, 8, 1, 8),  # All `True` values.
        ]
    )
    def test_finds_maximum_true_arg(
        self, sequence, min_arg, max_arg, allowance, expected,
    ):
        fn = lambda i: bool(sequence[i - 1])
        actual = metrics.maximum_true_arg(fn, min_arg, max_arg, allowance)
        assert expected == actual
