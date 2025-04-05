"""
Load EEG data from a text file into a pandas DataFrame.

Parameters:
file_path (str): Path to the input text file containing EEG data.

Returns:
pandas.DataFrame: DataFrame with columns as EEG channel labels and rows as data samples.
"""

import pandas as pd


class Neutronic:
    def __init__(self):
        # Define the EEG channel labels in the specified order
        self.channels = [
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

    def read_file(self, file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                # Ensure the line has enough elements to extract EEG channels
                if len(parts) >= 23:
                    eeg_values = parts[3:23]  # Extract the 20 EEG channel values
                    try:
                        # Convert to integers and add to data list
                        data_row = [int(val) for val in eeg_values]
                        data.append(data_row)
                    except ValueError:
                        # Skip lines with non-integer values in the EEG data
                        continue

        # Create DataFrame with the extracted data and channel labels
        return pd.DataFrame(data, columns=self.channels)
