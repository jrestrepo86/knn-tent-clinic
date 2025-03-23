import time

import pandas as pd
import ray
from data_sources.neutronic import Neutronic
from knn_tent.knn_tent import KnnTent
from preprocessing.tools import filter_signal, hilbert_phase, wavelet_phase
from ray.experimental.tqdm_ray import tqdm

NUM_CPUS = 4
WORKING_DIR = ""
ray.init(num_cpus=NUM_CPUS, runtime_env={"working_dir": WORKING_DIR})


@ray.remote
class Progress:
    def __init__(self, max_it=1, pbar=True):
        self.pbar_flag = pbar
        self.pbar = tqdm(total=max_it)
        self.count = -1
        self.max_it = max_it
        self.update()

    def update(self):
        if self.pbar_flag:
            self.pbar.update()
        else:
            self.count += 1
            p = 100 * self.count / self.max_it
            print(f"Progress: {p:.2f}%  {self.count}/{self.max_it}")


class EEGTent:
    def __init__(self, file, file_source="neutronic"):
        self.file = file
        self.reader = Neutronic()
        self.reader.read_file(file)
        self.raw_data = pd.DataFrame()
        self.raw_data = self.reader.data
        self.channels = self.reader.channels

    def _compute_tent(self, data, target_channel, source_channel, tent_parameters):
        m = tent_parameters["embeding_dim"]
        tau = tent_parameters["tau"]
        u = tent_parameters["u"]
        nn = tent_parameters["nn"]
        n_surrogates = tent_parameters["nsurrogates"]
        # Create KNN tent object
        target = data[target_channel]
        source = data[source_channel]

        knn_tent = KnnTent(target, source, m, tau, u, nn)
        # Compute transfer entropy
        out = knn_tent.knn_tent(n_surrogates=n_surrogates)
        return (target_channel, source_channel, out[0], out[1], out[2])

    @ray.remote()
    def _multiprocessing(self, data, target_channel, source_channel, tent_parameters):
        result = self._compute_tent(
            data, target_channel, source_channel, tent_parameters
        )
        if self.progress:
            self.progress.update.remote()
        return result

    def _get_phases(self, filter_parameters, type="hilbert"):
        lowcut = filter_parameters["lowcut"]
        highcut = filter_parameters["highcut"]
        fs = filter_parameters["fs"]
        order = filter_parameters["order"]

        # Compute phases
        self.phase_data = self.raw_data.copy()
        if type == "hilbert":
            for ch in self.channels:
                data = self.raw_data[ch]
                filter_data = filter_signal(data, lowcut, highcut, fs, order)
                self.phase_data[ch] = hilbert_phase(filter_data)
        elif type == "wavelet":
            for ch in self.channels:
                data = self.raw_data[ch]
                self.phase_data[ch] = wavelet_phase(
                    self.raw_data, lowcut, highcut, fs, omega0=5.0, num_scales=10
                )
        else:
            raise ValueError("Invalid type of phase")

    def _collect_reults(self, results_array):
        results = []
        results += [
            (target, source, Tent, Tent_no_sur, Tent_sur)
            for target, source, Tent, Tent_no_sur, Tent_sur in results_array
        ]
        results = pd.DataFrame(
            results, columns=["target", "source", "Tent", "Tent_no_sur", "Tent_sur"]
        )

    def tent(self, tent_parameters, filter_parameters, multiprocessing=False, log=True):

        self._get_phases(filter_parameters)

        DATA_id = ray.put(self.phase_data)
        # Set tent for each channel
        sims = []
        for target_channel in self.channels:
            for source_channel in self.channels:
                if target_channel != source_channel:
                    sim_params = {
                        "data": DATA_id,
                        "source": source_channel,
                        "target": target_channel,
                        "tent_parameters": tent_parameters,
                    }
                    sims.append(sim_params)

        res = []
        if multiprocessing and log:
            self.progress = Progress(max_it=len(sims))
        else:
            self.progress = []
        for s in sims:
            time.sleep(0.2)
            if multiprocessing:
                res.append(
                    self._multiprocessing.remote(
                        self, s["data"], s["target"], s["source"], s["tent_parameters"]
                    )
                )
            else:
                res.append(
                    self._compute_tent(
                        s["data"], s["target"], s["source"], s["tent_parameters"]
                    )
                )
        if multiprocessing:
            res = ray.get(res)
        return self._collect_reults(res)
