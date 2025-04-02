import itertools
import time

import pandas as pd
import ray
from config import ROOT_PATH
from data_sources.neutronic import Neutronic
from knn_tent.knn_tent import KnnTent
from preprocessing.tools import filter_signal, hilbert_phase, wavelet_phase

NUM_CPUS = 8
if not ray.is_initialized():
    ray.init(
        runtime_env={"working_dir": ROOT_PATH},
        ignore_reinit_error=True,
    )


class EEGTent:
    def __init__(self, file):
        self.file = file
        self.reader = Neutronic()
        self.reader.read_file(file)
        self.raw_data = pd.DataFrame()
        self.raw_data = self.reader.data
        self.channels = self.reader.channels

    def _compute_tent(self, data, target_channel, source_channel, tent_parameters):
        # Create KNN tent object
        target = data[target_channel].values
        source = data[source_channel].values

        knn_tent = KnnTent(
            source=source,
            target=target,
            m=tent_parameters["embedding_dim"],
            tau=tent_parameters["tau"],
            u=tent_parameters["u"],
            nn=tent_parameters["nn"],
        )
        # Compute transfer entropy
        out = knn_tent.knn_tent(n_surrogates=tent_parameters["nsurrogates"])
        return (source_channel, target_channel, out[0], out[1], out[2])

    @ray.remote(num_cpus=NUM_CPUS)
    def _multiprocessing(self, data, target_channel, source_channel, tent_parameters):
        result = self._compute_tent(
            data, target_channel, source_channel, tent_parameters
        )
        return result

    def _get_phases(self, filter_parameters, filter_type="hilbert"):
        lowcut = filter_parameters["lowcut"]
        highcut = filter_parameters["highcut"]
        fs = filter_parameters["fs"]
        order = filter_parameters["order"]

        # Compute phases
        self.phase_data = self.raw_data.copy()
        if filter_type == "hilbert":
            for ch in self.channels:
                data = self.raw_data[ch]
                filter_data = filter_signal(data, lowcut, highcut, fs, order)
                self.phase_data[ch] = hilbert_phase(filter_data)
        elif filter_type == "wavelet":
            for ch in self.channels:
                data = self.raw_data[ch]
                self.phase_data[ch] = wavelet_phase(
                    data, lowcut, highcut, fs, omega0=5.0, num_scales=10
                )
        else:
            raise ValueError("Invalid type of phase")

    def _substract_tent(self, results):
        pairs = list(itertools.combinations(self.channels, 2))
        results_ = []

        for ch1, ch2 in pairs:
            tent_ch1_ch2 = results[
                (results["source"] == ch1) & (results["target"] == ch2)
            ]["tent"].values[0]
            tent_ch2_ch1 = results[
                (results["source"] == ch2) & (results["target"] == ch1)
            ]["tent"].values[0]
            tent = tent_ch1_ch2 - tent_ch2_ch1
            flow = 1 if tent > 0 else -1
            results_.append((ch1, ch2, tent, flow))
        results_ = pd.DataFrame(results_, columns=["source", "target", "tent", "flow"])
        return results_

    def _get_flow(self, tent):
        return (tent > 0).astype("int")

    def tent(
        self,
        tent_parameters,
        filter_parameters,
        filter_type="hilbert",
        multiprocessing=False,
    ):

        self._get_phases(filter_parameters, filter_type=filter_type)

        if multiprocessing:
            data_id = ray.put(self.phase_data)
        else:
            data_id = self.raw_data
        # Set tent for each channel
        sims = []
        for target_channel in self.channels:
            for source_channel in self.channels:
                if target_channel != source_channel:
                    sim_params = {
                        "data": data_id,
                        "source": source_channel,
                        "target": target_channel,
                        "tent_parameters": tent_parameters,
                    }
                    sims.append(sim_params)

        res = []
        for s in sims:
            if multiprocessing:
                res.append(
                    self._multiprocessing.remote(
                        self, s["data"], s["source"], s["target"], s["tent_parameters"]
                    )
                )
            else:
                res.append(
                    self._compute_tent(
                        s["data"], s["source"], s["target"], s["tent_parameters"]
                    )
                )
        if multiprocessing:
            res = ray.get(res)

        results = list(
            (target, source, tent, tent_no_sur, tent_sur)
            for target, source, tent, tent_no_sur, tent_sur in res
        )
        results = pd.DataFrame(
            results, columns=["source", "target", "tent", "tent_no_sur", "tent_sur"]
        )
        results = self._substract_tent(results)

        return results
