import itertools
from pathlib import Path

import pandas as pd
import ray

from knn_tent_clinic import CUSTOM_FILTER_PARAMETERS, NUM_CPUS, ROOT_PATH
from knn_tent_clinic.data_sources.data_handler import DataHandler
from knn_tent_clinic.knn_tent.knn_tent import KnnTent
from knn_tent_clinic.preprocessing.tools import hilbert_phase, wavelet_phase

if not ray.is_initialized():
    ray.init(
        runtime_env={"working_dir": ROOT_PATH},
        ignore_reinit_error=True,
    )


class EEGTent:
    def __init__(self, file, sampling_frequency, hardware_source="neutronic"):
        self.source_file = Path(file)
        self.fs = sampling_frequency
        self.data_handler = DataHandler(self.source_file, hardware_source)
        self.raw_data = self.data_handler.read_data()
        self.channels = self.data_handler.get_channels()
        self.phase_data = pd.DataFrame()

    def _compute_tent(
        self, data, target_channel, source_channel, tent_parameters, freq_band
    ):
        # Create KNN tent object

        data = data[data["freq-band"] == freq_band]

        target = data[data["channel"] == target_channel]["data"].values[0]
        source = data[data["channel"] == source_channel]["data"].values[0]

        knn_tent = KnnTent(
            source=source,
            target=target,
            m=tent_parameters["embedding-dim"],
            tau=tent_parameters["tau"],
            u=tent_parameters["u"],
            nn=tent_parameters["nn"],
        )
        # Compute transfer entropy
        out = knn_tent.knn_tent(n_surrogates=tent_parameters["nsurrogates"])
        return (freq_band, source_channel, target_channel, out[0], out[1], out[2])

    @ray.remote(num_cpus=NUM_CPUS)
    def _multiprocessing(
        self, data, target_channel, source_channel, tent_parameters, freq_band
    ):
        result = self._compute_tent(
            data, target_channel, source_channel, tent_parameters, freq_band
        )
        return result

    def _get_phases(self, filters, fs, filter_type="hilbert"):
        filter_data_array = []
        for freq_band, filt in filters.items():
            lowcut = filt["lowcut"]
            highcut = filt["highcut"]
            order = filt["order"]
            for ch in self.channels:
                if filter_type == "hilbert":
                    filter_data = hilbert_phase(
                        self.raw_data[ch], lowcut, highcut, fs, order
                    )
                elif filter_type == "wavelet":
                    filter_data = wavelet_phase(
                        self.raw_data[ch],
                        lowcut,
                        highcut,
                        fs,
                        omega0=5.0,
                        num_scales=10,
                    )
                else:
                    raise ValueError("Invalid type of phase")
                filter_data_array.append((freq_band, ch, filter_data))
        self.phase_data = pd.DataFrame(
            filter_data_array, columns=["freq-band", "channel", "data"]
        )

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

    def tent(
        self,
        tent_parameters,
        filters_parameters=None,
        filter_type="hilbert",
        multiprocessing=False,
    ):

        if filters_parameters is None:
            self._get_phases(
                CUSTOM_FILTER_PARAMETERS, fs=self.fs, filter_type=filter_type
            )
        else:
            self._get_phases(filters_parameters, fs=self.fs, filter_type=filter_type)

        if multiprocessing:
            data_id = ray.put(self.phase_data)
        else:
            data_id = self.phase_data
        # Set tent for each channel
        freq_bands = self.phase_data["freq-band"].unique()
        sims = []
        for freq_band in freq_bands:
            for target_channel in self.channels:
                for source_channel in self.channels:
                    if target_channel != source_channel:
                        sim_params = {
                            "data": data_id,
                            "source": source_channel,
                            "target": target_channel,
                            "tent-parameters": tent_parameters,
                            "freq-band": freq_band,
                        }
                        sims.append(sim_params)

        res = []
        for s in sims:
            if multiprocessing:
                res.append(
                    self._multiprocessing.remote(
                        self,
                        s["data"],
                        s["source"],
                        s["target"],
                        s["tent-parameters"],
                        s["freq-band"],
                    )
                )
            else:
                res.append(
                    self._compute_tent(
                        s["data"],
                        s["source"],
                        s["target"],
                        s["tent-parameters"],
                        s["freq-band"],
                    )
                )
        if multiprocessing:
            res = ray.get(res)

        results = list(
            (freq_band, target, source, tent, tent_no_sur, tent_sur)
            for freq_band, target, source, tent, tent_no_sur, tent_sur in res
        )
        results = pd.DataFrame(
            results,
            columns=[
                "freq_band",
                "source",
                "target",
                "tent",
                "tent_no_sur",
                "tent_sur",
            ],
        )
        # results = self._substract_tent(results)

        return results
