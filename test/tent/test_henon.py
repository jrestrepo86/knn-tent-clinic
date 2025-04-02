"""Test Tent with hennon map"""

import sys

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm

try:
    from src.knn_tent.knn_tent import KnnTent
except ImportError:
    sys.path.append("../../src/")
    from knn_tent.knn_tent import KnnTent


def coupled_henon(n, c):
    """Coupled Henon map X1->X2"""
    n0 = 10000
    n = n + n0
    x = np.zeros((n, 2))
    x[0:2, :] = np.random.rand(2, 2)
    for i in range(2, n):
        x[i, 0] = 1.4 - x[i - 1, 0] ** 2 + 0.3 * x[i - 2, 0]

        x[i, 1] = (
            1.4
            - c * x[i - 1, 0] * x[i - 1, 1]
            - (1 - c) * x[i - 1, 1] ** 2
            + 0.3 * x[i - 2, 1]
        )

    x1, x2 = x[n0:, 0], x[n0:, 1]
    return x1, x2


def test_henon01():
    c = np.linspace(0, 0.9, 21)
    final_tent = []
    tent_sur = []

    for c0 in tqdm(c, desc="Processing coupled Henon map"):
        x1, x2 = coupled_henon(n=10000, c=c0)
        knn_x1_x2 = KnnTent(
            source=x1, target=x2, m=2, tau=3, u=1, nn=12, normalize_series=True
        )
        knn_x2_x1 = KnnTent(
            source=x2, target=x1, m=2, tau=3, u=1, nn=12, normalize_series=True
        )

        out_x1_x2 = knn_x1_x2.knn_tent(n_surrogates=10)
        out_x2_x1 = knn_x2_x1.knn_tent(n_surrogates=10)

        final_tent.append(out_x1_x2[0] - out_x2_x1[0])
        tent_sur.append(out_x1_x2[2] - out_x2_x1[2])

    fig = go.Figure()
    fig = go.Figure(go.Scatter(x=c, y=final_tent, name="Final_Tent"))
    fig.add_trace(go.Scatter(x=c, y=tent_sur, name="Tent_sur"))
    fig.update_layout(
        title="Tent  X1 -> X2",
    )
    fig.update_xaxes(title_text="Coupling")
    fig.update_yaxes(title_text="Tent")
    fig.show()


if __name__ == "__main__":
    test_henon01()
