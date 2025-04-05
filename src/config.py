import sys
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent

DATA_PATH = ROOT_PATH / "data"

NUM_CPUS = 8


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
