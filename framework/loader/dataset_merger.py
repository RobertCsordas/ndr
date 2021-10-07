import torch.utils.data
from typing import List, Any
import numpy as np
import bisect


class DatasetMerger(torch.utils.data.Dataset):
    def __init__(self, datasets: List[torch.utils.data.Dataset]):
        self.datasets = datasets
        self.lens = [len(d) for d in self.datasets]
        self.lsum = np.cumsum(self.lens).tolist()

    def __len__(self) -> int:
        return self.lsum[-1]

    def __getitem__(self, item: int) -> Any:
        ds_index = bisect.bisect(self.lsum, item)
        return self.datasets[ds_index][item - (self.lsum[ds_index-1] if ds_index>0 else 0)]
