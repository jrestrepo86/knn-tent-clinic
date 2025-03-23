"""
KNN-based estimation of Transfer Entropy (TE) using the Kraskov-Stögbauer-Grassberger (KSG) estimator.

This module implements the first algorithm from the KSG estimator to compute transfer entropy between two time series.

References:
1. https://doi.org/10.1103/PhysRevE.69.066138
2. arXiv:1411.2003
3. DOI: 10.1007/978-3-642-54474-3_1
"""

import numpy as np
from scipy.special import psi
from sklearn.neighbors import KDTree

from knn_tent.knn_tent_tools import (embedding, random_circular_shift,
                                     series_normalization, to_col_vector)


class KnnTent:
    """
    A class to compute Transfer Entropy (TE) using the k-nearest neighbors (KNN) approach.

    Attributes:
        source (np.ndarray): The source time series.
        target (np.ndarray): The target time series.
        m (int): Embedding dimension.
        tau_source (int): Time delay for the source signal.
        tau_target (int): Time delay for the target signal.
        u (int): Prediction horizon.
        nn (int): Number of nearest neighbors.
    """

    def __init__(
        self,
        target,
        source,
        m,
        tau,
        u,
        nn,
        normalize_series=False,
    ):
        """
        Initialize the KnnTent class.

        Args:
            source (np.ndarray): Source time series.
            target (np.ndarray): Target time series.
            m (int): Embedding dimension.
            tau (int or list): Time delay. If int, same delay for source and target.
                              If list, [tau_source, tau_target].
            u (int): Prediction horizon.
            nn (int): Number of nearest neighbors.
            normalize_series (bool): If True, normalize the input series.
        """
        self._validate_inputs(target, source, m, tau, u, nn)
        if normalize_series:
            self.source = series_normalization(self.source)
            self.target = series_normalization(self.target)

    def _validate_inputs(self, source, target, m, tau, u, nn):
        """Validate the input parameters."""
        if m < 1:
            raise ValueError("Embedding dimension (m) must be a positive integer.")
        self.m = int(m)

        if isinstance(tau, list):
            if len(tau) != 2:
                raise ValueError("If tau is a list, it must contain two integers.")
            self.tau_source, self.tau_target = tau
        elif isinstance(tau, int):
            self.tau_source = self.tau_target = tau
        else:
            raise ValueError("tau must be an integer or a list of two integers.")

        if not isinstance(u, int) or u < 1:
            raise ValueError("Prediction horizon (u) must be a positive integer.")
        self.u = int(u)

        if not isinstance(nn, int) or nn < 1:
            raise ValueError(
                "Number of nearest neighbors (nn) must be a positive integer."
            )
        self.nn = int(nn)

        if len(source) != len(target):
            raise ValueError("Source and target time series must have the same length.")
        self.source = to_col_vector(source)
        self.target = to_col_vector(target)

    def _embed_signals(self, target, source):
        """
        Embed the source and target signals into a higher-dimensional space.

        Args:
            target (np.ndarray): Target time series.
            source (np.ndarray): Source time series.

        Returns:
            np.ndarray: Embedded vector space.
        """
        target_embedded = embedding(target, self.m, self.tau_target)
        source_embedded = embedding(source, self.m, self.tau_source)
        N = min(source_embedded.shape[0], target_embedded.shape[0])

        # Truncate to the same length
        source_embedded = source_embedded[:N, :]
        target_embedded = target_embedded[:N, :]

        # Form the vector space: {Target_(t+u), Target_t, Source_t}
        embedded_space = np.concatenate(
            (
                target_embedded[self.u :, :],
                target_embedded[: -self.u, :],
                source_embedded[: -self.u, :],
            ),
            axis=1,
        )
        return embedded_space

    def _get_tent(self, target, source):
        """
        Compute the Transfer Entropy (TE) using the KSG estimator.

        Args:
            target (np.ndarray): Target time series.
            source (np.ndarray): Source time series.

        Returns:
            float: Transfer Entropy value.
        """
        metric = "chebyshev"
        embedded_space = self._embed_signals(source, target)
        N = embedded_space.shape[0]

        # Find nearest neighbors in the full embedding space
        tree = KDTree(embedded_space, metric=metric)
        nn_indices = tree.query(
            embedded_space, k=self.nn + 1, return_distance=False, sort_results=True
        )

        # Compute the maximum distance to the k-th nearest neighbor
        distances = (
            np.array(
                [
                    np.max(
                        np.abs(embedded_space[nn_indices[i]] - embedded_space[i]),
                        axis=1,
                    ).max()
                    for i in range(N)
                ]
            )
            - 1e-15  # Small offset to avoid numerical issues
        )

        # KSG estimation for different subspaces
        m = self.m
        subspaces = [
            (m, 2 * m),  # Target_t
            (0, 2 * m),  # Target_ut
            (m, 3 * m),  # TargetSource_t
        ]
        counts = []
        for start, end in subspaces:
            subspace_data = embedded_space[:, start:end]
            tree_sub = KDTree(subspace_data, metric=metric)
            count = tree_sub.query_radius(subspace_data, distances, count_only=True)
            counts.append(count)

        nnT, nnTu, nnTS = counts
        transfer_entropy = (
            psi(self.nn) + np.mean(psi(nnT)) - np.mean(psi(nnTu)) - np.mean(psi(nnTS))
        )

        return transfer_entropy

    def _get_tent02(self, target, source):
        metric = "chebyshev"
        vec = self._embed_signals(source, target)
        N = vec.shape[0]

        # Full embedding space neighbors
        tree = KDTree(vec, metric=metric)
        nn_inds = tree.query(
            vec, k=self.nn + 1, return_distance=False, sort_results=True
        )
        dist = (
            np.array(
                [
                    np.max(np.abs(vec[nn_inds[i]] - vec[i]), axis=1).max()
                    for i in range(N)
                ]
            )
            - 1e-15
        )

        # Original KSG estimation
        m = self.m
        subspaces = [
            (m, 2 * m),  # Target_t
            (0, 2 * m),  # Target_ut
            (m, 3 * m),  # TargetSource_t
        ]
        counts = []
        for start, end in subspaces:
            subspace_data = vec[:, start:end]
            tree_sub = KDTree(subspace_data, metric=metric)
            count = tree_sub.query_radius(subspace_data, dist, count_only=True)
            counts.append(count)

        nnT, nnTu, nnTS = counts
        Tksg = (
            psi(self.nn) + np.mean(psi(nnT)) - np.mean(psi(nnTu)) - np.mean(psi(nnTS))
        )

        return Tksg

    def _get_surrogates(self, target, source, n_surrogates):
        """
        Compute surrogate Transfer Entropy values by random circular shifting.

        Args:
            target (np.ndarray): Target time series.
            source (np.ndarray): Source time series.
            n_surrogates (int): Number of surrogate samples.

        Returns:
            float: Mean surrogate Transfer Entropy.
        """
        surrogates = []
        for _ in range(n_surrogates):
            surrogate_source = random_circular_shift(source)
            surrogates.append(self._get_tent(target, surrogate_source))
        return np.array(surrogates).mean()

    def knn_tent(self, n_surrogates=30):
        """
        Compute the Transfer Entropy and its surrogate values.

        Returns:
            tuple: (Transfer Entropy, Surrogate Transfer Entropy)
        """
        T_target_source = self._get_tent(self.target, self.source)
        T_source_target = self._get_tent(self.source, self.target)
        T_ts_surrogates = self._get_surrogates(
            self.target, self.source, n_surrogates=n_surrogates
        )
        T_st_surrogates = self._get_surrogates(
            self.source, self.target, n_surrogates=n_surrogates
        )

        Tent = T_target_source - T_source_target
        Tent_surrogates = T_ts_surrogates - T_st_surrogates
        return Tent - Tent_surrogates, Tent, Tent_surrogates
