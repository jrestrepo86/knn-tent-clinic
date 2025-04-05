import ray
from ray.experimental.tqdm_ray import tqdm

CUSTOM_FILTER_PARAMETERS = {
    "alpha": {
        "lowcut": 8.0,
        "highcut": 12.0,
        "order": 5,
    },
    "beta1": {
        "lowcut": 19.0,
        "highcut": 25.0,
        "order": 5,
    },
    "beta2": {
        "lowcut": 25.0,
        "highcut": 31.0,
        "order": 5,
    },
    "tetha": {
        "lowcut": 4.0,
        "highcut": 8.0,
        "order": 5,
    },
}


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
