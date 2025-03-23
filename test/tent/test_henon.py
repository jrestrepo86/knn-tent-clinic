import sys

import numpy as np
import plotly.graph_objects as go

try:
    from src.knn_tent.knn_tent import KnnTent
except ImportError:
    sys.path.append("../../src/")
    from knn_tent.knn_tent import KnnTent


def coupledHenon(n, c):
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

    source, target = x[n0:, 0], x[n0:, 1]
    return source, target


def test_henon01():
    c = np.linspace(0, 0.9, 21)
    Final_Tent = []
    Tent = []
    Tent_sur = []
    for c0 in c:
        source, target = coupledHenon(n=10000, c=c0)
        knn_tent = KnnTent(
            target, source, m=2, tau=3, u=1, nn=12, normalize_series=True
        )
        out = knn_tent.knn_tent(n_surrogates=10)
        Final_Tent.append(out[0])
        Tent.append(out[1])
        Tent_sur.append(out[2])

    fig = go.Figure()
    fig = go.Figure(go.Scatter(x=c, y=Final_Tent, name="Final_Tent"))
    fig.add_trace(go.Scatter(x=c, y=Tent, name="Tent"))
    fig.add_trace(go.Scatter(x=c, y=Tent_sur, name="Tent_sur"))
    fig.show()


if __name__ == "__main__":
    test_henon01()
