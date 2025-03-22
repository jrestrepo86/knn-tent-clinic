import numpy as np
from numpy.lib.stride_tricks import as_strided

from sklearn.decomposition import PCA  # Add this import
from scipy.special import gamma  # Alternative for array inputs


def toColVector(x):
    """
    Change vectors to column vectors
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def series_normalization(x):
    """
    Add normal noise  to the series and normalize each  column to cero mean
    and unitary variance
    """
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = x + 1e-10 * np.random.rand(*x.shape)
    return x


def embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Efficient Takens embedding for single-channel signals using stride tricks.

    Args:
        x: Input signal (1D array or column vector)
        m: Embedding dimension (number of delay coordinates)
        tau: Time delay (number of samples between coordinates)

    Returns:
        2D array of shape (N_emb, m) containing embedded vectors

    Raises:
        ValueError: For invalid input dimensions or parameters
    """
    # Convert to column vector and validate input

    n_samples = x.shape[0]
    min_length = (m - 1) * tau + 1

    if n_samples < min_length:
        raise ValueError(
            f"Signal too short: Need {min_length} samples, got {n_samples}"
        )

    # Create memory-efficient strided view
    itemsize = x.strides[0]
    strides = (
        itemsize,  # Step between embedded vectors
        tau * itemsize,
    )  # Step between coordinates

    embedded = as_strided(
        x, shape=(n_samples - (m - 1) * tau, m), strides=strides, writeable=False
    )

    return embedded.copy()  # Return contiguous array for safe usage


def correct_volume(neighbors, data_min, data_max, metric):
    """
    Compute original and corrected log-volumes for kNN-based transfer entropy.

    Implements the boundary correction from Gao et al. (2015) for KSG estimators.

    Args:
        neighbors: Array of shape (k+1, d) containing:
                   - query point (first row)
                   - k nearest neighbors
        data_min: Global minima for each dimension (1D array)
        data_max: Global maxima for each dimension (1D array)
        metric: Distance metric ('chebyshev' for L∞ or 'euclidean')

    Returns:
        log_vol: Original log-volume
        log_vol_rec: Corrected log-volume after boundary adjustment
    """
    # Extract query point and neighbors
    query_point = neighbors[0]
    neighbor_points = neighbors[1:]

    # Calculate maximum distance to k-th neighbor
    if metric == "chebyshev":
        distances = np.abs(neighbor_points - query_point)
        max_distance = np.max(distances)
    else:
        # euclidean metric:
        max_distance = np.max(np.linalg.norm(neighbor_points - query_point, axis=1))

    # Calculate original volume
    d = query_point.shape[0]
    if metric == "chebyshev":
        log_vol = d * np.log(2 * max_distance)
    else:  # Euclidean
        log_vol = (
            (d / 2) * np.log(np.pi)
            - np.log(gamma(d / 2 + 1))
            + d * np.log(max_distance)
        )

    # Calculate available space in each dimension
    available_lengths = np.empty(d)
    for j in range(d):
        lower_bound = max(query_point[j] - max_distance, data_min[j])
        upper_bound = min(query_point[j] + max_distance, data_max[j])
        available_lengths[j] = upper_bound - lower_bound

    # Handle zero-volume dimensions
    np.clip(available_lengths, 1e-100, None, out=available_lengths)

    # Calculate corrected volume
    log_vol_rec = np.sum(np.log(available_lengths))

    # For Euclidean: adjust sphere volume to match data bounds
    if metric == "euclidean":
        sphere_correction = np.log(gamma(d / 2 + 1)) - (d / 2) * np.log(np.pi)
        log_vol_rec += sphere_correction - d * np.log(2)

    return log_vol, log_vol_rec
