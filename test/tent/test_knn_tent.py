import pytest
import numpy as np
from src.knn_tent.knn_tent import KnnTent
from src.knn_tent.knn_tent_tools import embedding, correct_volume


def test_basic_functionality():
    """Test if the algorithm runs and returns a float."""
    np.random.seed(42)
    source = np.random.randn(100)
    target = np.random.randn(100)
    kt = KnnTent(source, target, m=2, tau=1, u=1, nn=3)
    te_value = kt.knn_tent_alg01(vcor=True)
    assert isinstance(te_value, float)


def test_input_validation():
    """Test invalid parameter handling."""
    source = np.random.randn(10)
    target = np.random.randn(10)

    # Invalid m
    with pytest.raises(ValueError):
        KnnTent(source, target, m=0, tau=1, u=1, nn=3)

    # Invalid tau type
    with pytest.raises(ValueError):
        KnnTent(source, target, m=2, tau="invalid", u=1, nn=3)

    # Invalid u
    with pytest.raises(ValueError):
        KnnTent(source, target, m=2, tau=1, u=0, nn=3)

    # Mismatched lengths
    with pytest.raises(ValueError):
        KnnTent(source[:-1], target, m=2, tau=1, u=1, nn=3)


def test_normalization():
    """Test series normalization."""
    np.random.seed(42)
    source = np.random.randn(100)
    target = np.random.randn(100)
    kt = KnnTent(source, target, m=2, tau=1, u=1, nn=3, normalize_series=True)
    assert np.allclose(np.mean(kt.source, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(kt.source, axis=0), 1, atol=1e-7)
    assert np.allclose(np.mean(kt.target, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(kt.target, axis=0), 1, atol=1e-7)


def test_embedding():
    """Test Takens embedding."""
    x = np.arange(10)
    embedded = embedding(x, m=3, tau=2)
    expected = np.array(
        [[0, 2, 4], [1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8], [5, 7, 9]]
    )
    assert np.array_equal(embedded, expected)

    # Test short input
    with pytest.raises(ValueError):
        embedding(np.array([1, 2, 3]), m=3, tau=2)


def test_correct_volume_chebyshev():
    """Test volume correction for Chebyshev metric."""
    neighbors = np.array([[0.1, 0.1], [0.3, 0.3], [0.0, 0.0]])
    data_min = np.array([0.0, 0.0])
    data_max = np.array([1.0, 1.0])
    metric = "chebyshev"
    log_vol, log_vol_rec = correct_volume(neighbors, data_min, data_max, metric)

    max_distance = 0.2  # Max Chebyshev distance from query point (0.1) to neighbors
    expected_log_vol = 2 * np.log(2 * max_distance)
    expected_log_vol_rec = 2 * np.log(0.3 - 0.0)  # 0.3 upper - 0.0 lower

    assert np.isclose(log_vol, expected_log_vol)
    assert np.isclose(log_vol_rec, expected_log_vol_rec)


def test_identical_source_target():
    """Test TE when source and target are identical."""
    np.random.seed(42)
    data = np.random.randn(100)
    kt = KnnTent(data, data, m=2, tau=1, u=1, nn=3, normalize_series=True)
    te_value = kt.knn_tent_alg01(vcor=True)
    assert te_value >= 0  # TE should be non-negative


def test_vcor_flag():
    """Test with and without volume correction."""
    np.random.seed(42)
    source = np.random.randn(100)
    target = np.random.randn(100)
    kt = KnnTent(source, target, m=2, tau=1, u=1, nn=3)
    te_vcor = kt.knn_tent_alg01(vcor=True)
    te_no_vcor = kt.knn_tent_alg01(vcor=False)
    assert te_vcor != te_no_vcor
