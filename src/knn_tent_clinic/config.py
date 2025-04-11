from pathlib import Path

# Define root path
ROOT_PATH = Path(__file__).parent


# Define data path
DATA_PATH = ROOT_PATH / "data"

NUM_CPUS = 8

CUSTOM_FILTER_PARAMETERS = {
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

MATRIX_CHANNELS_ORDER = [
    "F1",
    "F3",
    "C3",
    "P3",
    "O1",
    "F7",
    "T3",
    "T5",
    "Fz",
    "Cz",
    "Pz",
    "Oz",
    "T6",
    "T4",
    "F8",
    "O2",
    "P4",
    "C4",
    "F4",
    "F2",
]
