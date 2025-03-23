import numpy as np
from numpy.lib.stride_tricks import as_strided


def to_col_vector(x):
    """
    Convert a 1D array to a column vector.

    Args:
        x (np.ndarray): Input array.

    Returns:
        np.ndarray: Column vector.
    """
    x = x.reshape(x.shape[0], -1)
    if x.shape[0] < x.shape[1]:
        x = x.T
    return x


def series_normalization(x):
    """
    Normalize the input series to zero mean and unit variance, then add small noise.

    Args:
        x (np.ndarray): Input series.

    Returns:
        np.ndarray: Normalized series with added noise.
    """
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    x = x + 1e-10 * np.random.rand(*x.shape)
    return x


def embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """
    Perform Takens embedding for single-channel signals using stride tricks.

    Args:
        x (np.ndarray): Input signal (1D array or column vector).
        m (int): Embedding dimension (number of delay coordinates).
        tau (int): Time delay (number of samples between coordinates).

    Returns:
        np.ndarray: 2D array of shape (N_emb, m) containing embedded vectors.

    Raises:
        ValueError: If the signal is too short for the given embedding parameters.
    """
    n_samples = x.shape[0]
    min_length = (m - 1) * tau + 1

    if n_samples < min_length:
        raise ValueError(
            f"Signal too short: Need {min_length} samples, got {n_samples}"
        )

    # Create a memory-efficient strided view
    itemsize = x.strides[0]
    strides = (
        itemsize,  # Step between embedded vectors
        tau * itemsize,  # Step between coordinates
    )

    embedded = as_strided(
        x, shape=(n_samples - (m - 1) * tau, m), strides=strides, writeable=False
    )

    return embedded.copy()  # Return a contiguous array for safe usage


def random_circular_shift(x):
    """
    Perform a random circular shift on the input signal.

    Args:
        x (np.ndarray): Input signal.

    Returns:
        np.ndarray: Circularly shifted signal.
    """
    shift_amount = np.random.randint(0, len(x))
    shifted_signal = np.roll(x, shift_amount)
    return shifted_signal
