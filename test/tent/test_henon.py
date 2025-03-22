import numpy as np
import sys
import plotly.graph_objects as go

try:
    from src.knn_tent.knn_tent import KnnTent
except ImportError:
    sys.path.append("../src/")
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
    Txy = []
    Tyx = []
    for c0 in c:
        x, y = coupledHenon(n=10000, c=c0)
        Ktxy = KnnTent(x, y, m=2, tau=3, u=1, nn=12)
        Ktyx = KnnTent(y, x, m=2, tau=3, u=1, nn=12)
        Txy.append(Ktxy.knn_tent())
        Tyx.append(Ktyx.knn_tent())
    Txy = np.array(Txy)
    Tyx = np.array(Tyx)

    fig = go.Figure()
    fig = go.Figure(go.Scatter(x=c, y=Txy, name="Txy"))
    fig.add_trace(go.Scatter(x=c, y=Tyx, name="Tyx"))
    fig.show()


if __name__ == "__main__":
    test_henon01()
