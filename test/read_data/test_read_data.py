import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys


ROOT_PATH = Path(__file__).absolute().parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

try:
    from src.preprocessing.neutronic import Neutronic
except ImportError:
    sys.path.append("../../src/")
    from preprocessing.neutronic import Neutronic


def test_neutronic():
    file = DATA_PATH / "neutronic_data.txt"
    Reader = Neutronic()
    Reader.read_file(file)
    print(Reader.data)


if __name__ == "__main__":
    test_neutronic()
