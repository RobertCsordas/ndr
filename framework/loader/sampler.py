import torch
import torch.utils.data
from .. import utils
import numpy as np
import math
from typing import Dict, Any, List, Optional


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, replacement=True, seed=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self.seed = utils.seed.get_randstate(seed)

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            while True:
                yield self.seed.randint(0, n, dtype=np.int64)
        else:
            i_list = None
            pos = n
            while True:
                if pos >= n:
                    i_list = self.seed.permutation(n).tolist()
                    pos = 0

                sample = i_list[pos]
                pos += 1
                yield sample

    def __len__(self):
        return 0x7FFFFFFF


class FixedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.order = utils.seed.get_randstate(0xB0C1FA53).permutation(len(self.data_source)).tolist()

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return len(self.data_source)


class SubsetSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, n_max: int):
        super().__init__(data_source)
        self.data_source = data_source
        self._len = min(len(self.data_source), n_max)
        self.order = utils.seed.get_randstate(0xB0C1FA53).choice(len(self.data_source), self._len, replace=False)

    def __iter__(self):
        for i in self.order:
            yield i

    def __len__(self):
        return self._len


class MultibatchSequentialSampler:
    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int):
        self.ds_len = len(data_source)
        self.batch_size = batch_size
        self.len = self.ds_len // self.batch_size
        self.pos = None

    def __iter__(self):
        if self.pos is None:
            self.pos = 0

        while self.pos < self.len:
            p = self.pos
            self.pos += 1
            yield [b * self.len + p for b in range(self.batch_size)]

        self.pos = None

    def __len__(self):
        return self.len

    def state_dict(self) -> Dict[str, Any]:
        return {"pos": self.pos}

    def load_state_dict(self, state: Dict[str, Any]):
        self.pos = state["pos"]


class BucketedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int, length_key_name: str = "in_len",
                 infinite: bool = False, seed: Optional[int] = None, drop_last: bool = False, long_first: bool = False,
                 random_order: bool = True):
        super().__init__(data_source)
        self.lens = [data_source[i][length_key_name] for i in range(len(data_source))]
        self.batch_size = batch_size
        self.seed = utils.seed.get_randstate(seed)
        self.infinite = infinite
        self.drop_last = drop_last
        self.reverse = long_first
        self.random_order = random_order
        assert (not long_first) or (not self.random_order)

    def makebins(self) -> List[List[int]]:
        # First shuffle all
        order = self.seed.permutation(len(self.lens)).tolist()
        if self.drop_last:
            order = order[:-(len(order) % self.batch_size)]
        # Sort preverses the order of the same-length elements, thus the previous shuffle makes random elements to be
        # binned together
        order = list(sorted(order, key=lambda i: self.lens[i], reverse=self.reverse))
        return [order[i: i + self.batch_size] for i in range(0, len(order), self.batch_size)]

    def __iter__(self):
        while True:
            batches = self.makebins()

            t = self.seed.permutation if self.random_order else range
            for o in t(len(batches)):
                yield batches[o]

            if not self.infinite:
                break

    def __len__(self):
        return math.ceil(len(self.lens) / self.batch_size)
