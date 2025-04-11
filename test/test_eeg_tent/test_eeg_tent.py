"""Test eeg"""

import pandas as pd

from knn_tent_clinic import DATA_PATH
from knn_tent_clinic.clinic_tent.eeg_tent import EEGTent
from knn_tent_clinic.data_visualization.matrix_plot import make_matrices_plot


def test_eeg_tent():
    file = DATA_PATH / "neutronic_data.txt"
    tent_parameters = {
        "embedding-dim": 2,
        "tau": 2,
        "u": 2,
        "nn": 10,
        "nsurrogates": 5,
    }

    filters = {
        "theta": {
            "lowcut": 4.0,
            "highcut": 8.0,
            "order": 4,
        },
        "alpha": {
            "lowcut": 8.0,
            "highcut": 12.0,
            "order": 4,
        },
        "beta1": {
            "lowcut": 19.0,
            "highcut": 25.0,
            "order": 4,
        },
        "beta2": {
            "lowcut": 25.0,
            "highcut": 31.0,
            "order": 4,
        },
    }

    eeg_tent = EEGTent(file, sampling_frequency=65)
    results = eeg_tent.tent(
        tent_parameters,
        filters=filters,
        filter_type="hilbert",
        multiprocessing=True,
    )
    results.to_pickle("./data.pkl")
    fig = make_matrices_plot(results)
    fig.show()


if __name__ == "__main__":
    test_eeg_tent()
