import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT_PATH = Path(__file__).absolute().parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

try:
    from src.clinic_tent.eeg_tent import EEGTent
    from src.data_visualization.matrix_plot import flow_matrix_plot, tent_matrix_plot
except ImportError:
    sys.path.append("../../src/")
    from clinic_tent.eeg_tent import EEGTent
    from data_visualization.matrix_plot import flow_matrix_plot, tent_matrix_plot


def test_eeg_tent():
    file = DATA_PATH / "neutronic_data.txt"
    tent_parameters = {
        "embedding_dim": 2,
        "tau": 2,
        "u": 1,
        "nn": 8,
        "nsurrogates": 1,
    }
    filter_parameters = {"lowcut": 8, "highcut": 12, "fs": 65, "order": 4}
    eeg_tent = EEGTent(file)
    results = eeg_tent.tent(
        tent_parameters,
        filter_parameters,
        filter_type="hilbert",
        multiprocessing=True,
    )
    # Create a tent matrix plot
    tent_fig = tent_matrix_plot(results)
    # Create a flow matrix plot
    flow_fig = flow_matrix_plot(results)

    # Show the plot
    tent_fig.show()
    flow_fig.show()


if __name__ == "__main__":
    test_eeg_tent()
