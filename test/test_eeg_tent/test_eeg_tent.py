"""Test eeg"""

from knn_tent_clinic import DATA_PATH
from knn_tent_clinic.clinic_tent.eeg_tent import EEGTent
from knn_tent_clinic.data_visualization.matrix_plot import make_matrices_plots


def test_eeg_tent():
    file = DATA_PATH / "neutronic_data.txt"
    tent_parameters = {
        "embedding-dim": 2,
        "tau": 2,
        "u": 1,
        "nn": 5,
        "nsurrogates": 5,
    }
    eeg_tent = EEGTent(file, sampling_frequency=65)
    results = eeg_tent.tent(
        tent_parameters,
        filter_type="hilbert",
        multiprocessing=True,
    )
    fig = make_matrices_plots(results)
    fig.show()
    # # Create a tent matrix plot
    # tent_fig = tent_matrix_plot(results)
    # # Create a flow matrix plot
    # flow_fig = flow_matrix_plot(results)
    #
    # # Show the plot
    # tent_fig.show()
    # flow_fig.show()


if __name__ == "__main__":
    test_eeg_tent()
