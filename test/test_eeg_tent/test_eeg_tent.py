import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT_PATH = Path(__file__).absolute().parent.parent.parent
DATA_PATH = ROOT_PATH / "data"

try:
    from src.clinic_tent.eeg_tent import EEGTent
except ImportError:
    sys.path.append("../../src/")
    from clinic_tent.eeg_tent import EEGTent


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
        log=True,
    )
    # Pivot the DataFrame to create a matrix
    pivot_df = results.pivot(index="target", columns="source", values="Tent")

    # Create the heatmap using Plotly
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Source", y="Target", color="Tent"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Viridis",
    )

    # Update layout for better visualization
    fig.update_layout(
        title="Tent Matrix: Target vs Source",
        xaxis_nticks=len(pivot_df.columns),
        yaxis_nticks=len(pivot_df.index),
    )

    # Show the plot
    fig.show()


if __name__ == "__main__":
    test_eeg_tent()
