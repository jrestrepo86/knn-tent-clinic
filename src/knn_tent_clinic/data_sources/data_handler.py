from pathlib import Path

from .neutronic import Neutronic
from .open_bci import OpenBci


class DataHandler:
    def __init__(self, source_file, hardware_source):
        self.source = hardware_source
        self.source_file = Path(source_file)
        if hardware_source == "neutronic":
            self.data_handler = Neutronic()
        else:
            self.data_handler = OpenBci()

    def read_data(self):
        return self.data_handler.read_file(self.source_file)

    def get_channels(self):
        return self.data_handler.channels
