"""
KNNTent k-nearest neighbor estimation of transfer entropy
 This  algorithm  is  based  on  the  Kraskov-Stögbauer-Grassberger estimatiors (first
 algorithm)

Parameters:
    source: source signal.
    target: target signal.
    m: embedding dimension.
    tau: embedding lag. int or List, if tau is an int then tauSource = tauTarget = tau.
         if tau is a list (len(tau) = 2) then tauSource = tau[0] and tauTarget = tau[1].
    u: time ahead.
    nn: number of nearest-neighbors.
    snorm: if True normalize input signals to zero mean and unitary std, then add small
           amplitude noise.

Return:
    Tksg: Transfer entropy from source to target.

 refs:
 1. https://doi.org/10.1103/PhysRevE.69.066138
 2. arXiv:1411.2003
 3. DOI: 10.1007/978-3-642-54474-3_1
 4. The  transfer  entropy  estimator  for  the  SECOND  ksg-algorithm  was  found  in
 10.1007/978-3-642-54474-3_1.

"""

import numpy as np
from scipy.special import psi
from sklearn.neighbors import KDTree

from knn_tent.knn_tent_tools import (
    toColVector,
    series_normalization,
    embedding,
)


class KnnTent:
    def __init__(
        self,
        source,
        target,
        m,
        tau,
        u,
        nn,
        normalize_series=False,
    ):
        self._validate_inputs(source, target, m, tau, u, nn)
        if normalize_series:
            self.source = series_normalization(self.source)
            self.target = series_normalization(self.target)

    def _validate_inputs(self, source, target, m, tau, u, nn):
        """Validate input parameters."""
        if m < 1:
            raise ValueError("m must be a positive integer.")
        else:
            self.m = int(m)
        if not (
            isinstance(tau, (int, list)) or (isinstance(tau, list) and len(tau) != 2)
        ):
            raise ValueError("tau must be an int or a list of two integers.")
        else:
            if isinstance(tau, list):
                self.tau_source = tau[0]
                self.tau_target = tau[1]
            else:
                self.tau_source = self.tau_target = tau
        if not isinstance(u, int) or u < 1:
            raise ValueError("u must be a positive integer.")
        else:
            self.u = int(u)
        if not isinstance(nn, int) or nn < 1:
            raise ValueError("nn must be a positive integer.")
        else:
            self.nn = int(nn)
        if len(source) != len(target):
            raise ValueError("Source and target must have the same length.")
        else:
            self.source = toColVector(source)
            self.target = toColVector(target)

    def _embed_signals(self, source, target):
        sv = embedding(source, self.m, self.tau_source)
        tv = embedding(target, self.m, self.tau_target)
        N = min(sv.shape[0], tv.shape[0])
        sv = sv[:N, :]
        tv = tv[:N, :]
        # Form vector space vec = {Target_(t+u), Target_t, Source_t}
        vec = np.concatenate(
            (tv[self.u :, :], tv[: -self.u, :], sv[: -self.u, :]), axis=1
        )
        return vec

    def knn_tent(self):
        metric = "chebyshev"
        vec = self._embed_signals(self.source, self.target)
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
