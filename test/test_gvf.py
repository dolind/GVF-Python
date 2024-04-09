import numpy as np
import pytest
from gvf.GVF2D import BoundMirrorShrink, BoundMirrorExpand, \
    BoundMirrorEnsure  # Adjust this import based on your project structure

def test_bound_mirror_expand():
    A = np.array([
        [1, 2, 3, 11],
        [4, 5, 6, 12],
        [7, 8, 9, 13],
    ])
    expected = np.array([
        [5, 4, 5, 6, 12, 6],
        [2, 1, 2, 3, 11, 3],
        [5, 4, 5, 6, 12, 6],
        [8, 7, 8, 9, 13, 9],
        [5, 4, 5, 6, 12, 6],
    ])
    result = BoundMirrorExpand(A)
    np.testing.assert_array_equal(result, expected)

def test_bound_mirror_shrink():
    A = np.array([
        [5, 4, 5, 6, 12, 6],
        [2, 1, 2, 3, 11, 3],
        [5, 4, 5, 6, 12, 6],
        [8, 7, 8, 9, 13, 9],
        [5, 4, 5, 6, 12, 6],
    ])
    expected = np.array([
        [1, 2, 3, 11],
        [4, 5, 6, 12],
        [7, 8, 9, 13],
    ])
    result = BoundMirrorShrink(A)
    np.testing.assert_array_equal(result, expected)

def test_bound_mirror_ensure():
    A = np.array([
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, 1, 2, 3, 11, np.nan],
        [np.nan, 4, 5, 6, 12, np.nan],
        [np.nan, 7, 8, 9, 13, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    ])
    expected = np.array([
        [5, 4, 5, 6, 12, 6],
        [2, 1, 2, 3, 11, 3],
        [5, 4, 5, 6, 12, 6],
        [8, 7, 8, 9, 13, 9],
        [5, 4, 5, 6, 12, 6],
    ])
    result = BoundMirrorEnsure(A)
    np.testing.assert_array_equal(result, expected)

def test_bound_mirror_ensure_error():
    # Test with an array having less than 3 rows or columns
    A = np.array([[1, 2], [3, 4]])
    with pytest.raises(Exception):
        BoundMirrorEnsure(A)

def test_bound_mirror_shrink_4x3():
    # Input matrix A with an outer boundary that would be removed
    A = np.array([
        [9, 8, 7, 8],
        [6, 1, 2, 1],
        [5, 4, 3, 4],
        [6, 7, 8, 7]
    ])
    # Expected output matrix after shrinking
    expected = np.array([
        [1, 2],
        [4, 3]
    ])
    result = BoundMirrorShrink(A)
    np.testing.assert_array_equal(result, expected)

def test_bound_mirror_expand_4x3():
    # Original 4x3 matrix
    A = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12]
    ])
    # Expected result after expanding should be 6x5, with mirrored boundaries
    expected = np.array([
        [5, 4, 5, 6, 5],
        [2, 1, 2, 3, 2],
        [5, 4, 5, 6, 5],
        [8, 7, 8, 9, 8],
        [11, 10, 11, 12, 11],
        [8, 7, 8, 9, 8]
    ])
    result = BoundMirrorExpand(A)
    np.testing.assert_array_equal(result, expected)

def test_bound_mirror_ensure_4x3():
    # Input matrix with simulated non-mirrored boundary
    A = np.array([
        [99, 99, 99, 99, 99],  # Assume '99' values are placeholders and should be mirrored over
        [99, 1, 2, 3, 99],
        [99, 4, 5, 6, 99],
        [99, 7, 8, 9, 99],
        [99, 99, 99, 99, 99]
    ])
    # Expected output after ensuring mirror boundary
    expected = np.array([
        [5, 4, 5, 6, 5],  # Mirrored top boundary
        [2, 1, 2, 3, 2],  # Corrected left and right boundaries
        [5, 4, 5, 6, 5],
        [8, 7, 8, 9, 8],
        [5, 4, 5, 6, 5]   # Mirrored bottom boundary
    ])
    result = BoundMirrorEnsure(A)
    np.testing.assert_array_equal(result, expected)