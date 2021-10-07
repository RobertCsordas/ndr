from numpy.random import sample
import torch.utils.data
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from .helpers import rejection_sample_length_buckets, CacheLoaderMixin
import framework
from .sequence import TextClassifierTestState


@dataclass
class Sample:
    input: str
    output: int
    depth: int


@dataclass
class TranslatedSample:
    input: List[int]
    output: int
    depth: int


class SimpleArithmetics(CacheLoaderMixin, torch.utils.data.Dataset):
    OPS = [
        ("+", lambda a, b: a + b),
        ("*", lambda a, b: a * b)
    ]

    SPLIT_SEEDS = {"train": 0, "valid": 123}
    VERSION = 2
    N_NUMS = 10
    in_vocabulary = None
    out_vocabulary = None

    def config_id(self) -> str:
        return f"{self.split}_{self.n_digits}_{self.depth}_{self.p_subop}_{self.max_length}_{self.n_samples}_"\
               F"{self.rpn}_{self.n_nums}"

    def format_op(self, a1, a2, op):
        return f"{a1} {a2}{op}" if self.rpn else f"({a1}{op}{a2})"

    def get_one(self, seed, min_depth: int, max_depth: int) -> Sample:
        def sample_single_op(a1: Sample, a2: Sample) -> Sample:
            op = self.OPS[seed.randint(0, len(self.OPS))]
            expr = self.format_op(a1.input, a2.input, op[0])
            return Sample(expr, op[1](a1.output, a2.output) % self.maxval, max(a1.depth, a2.depth) + 1)

        def get(d: int) -> Sample:
            if (d >= max_depth) or (seed.rand() > self.p_subop):
                res = seed.randint(0, self.maxval)
                assert res  < self.maxval
                return Sample(str(res), res, 1)
            else:
                a1 = get(d + 1)
                a2 = get(d + 1)
                return sample_single_op(a1, a2)

        s = get(1)
        while s.depth < min_depth:
            other = get(2)
            s, other = (s, other) if seed.rand() < 0 else (other, s)
            s = sample_single_op(s, other)

        return s

    def generate_basic_set(self) -> List[Sample]:
        res = []
        for op in self.OPS:
            for a1 in range(self.maxval):
                for a2 in range(self.maxval):
                    res.append(Sample(self.format_op(a1, a2, op[0]), op[1](a1, a2) % self.maxval, 1))
        return res

    def generate_balanced_set(self, depth: Tuple[int, int], seed, n_samples: int, exclude=set()) -> List[Sample]:
        def get_sample(seed: np.random.RandomState):
            while True:
                s = self.get_one(seed, *depth)
                if len(s.input) <= self.max_length:
                    return s

        return rejection_sample_length_buckets(n_samples, depth, seed, get_sample, lambda s: s.depth,
                                               lambda s: hash(s.input), exclude, limits=self.limits_per_depth)

    def translate_sample(self, s: Sample) -> TranslatedSample:
        return TranslatedSample(self.in_vocabulary(s.input), self.out_vocabulary([str(s.output)])[0], s.depth)

    def generate_dataset(self) -> List[Sample]:
        seed = np.random.RandomState(self.SPLIT_SEEDS[self.split])

        res = []
        if self.depth[0] == 2:
            res += self.generate_basic_set()

        if self.depth[1] > 2:
            res += self.generate_balanced_set((max(self.depth[0], 3), self.depth[1]), seed, self.n_samples)

        return [self.translate_sample(s) for s in res]

    def construct_vocab(self):
        if self.in_vocabulary is None:
            digits = [str(i) for i in range(self.maxval)]
            SimpleArithmetics.in_vocabulary = framework.data_structures.WordVocabulary(["(", ")"] + [o[0] for o in self.OPS] + digits)
            SimpleArithmetics.out_vocabulary = framework.data_structures.WordVocabulary(digits)


    def __init__(self, split: str, n_digits: int, depth: Tuple[int, int], p_subop: float, max_length: int,
                 n_samples: int, rpn: bool = False, n_nums: Optional[int] = None):

        def n_possible(depth: int) -> int:
            def n(depth: int) -> int:
                return 0 if depth == 0 else (self.maxval + len(self.OPS) * n(depth - 1)**2)

            return n(depth) - n(depth - 1)

        super().__init__()

        assert 1 < depth[0] <= depth[1]

        self.rpn = rpn
        self.n_digits, self.depth, self.p_subop, self.max_length, self.split, self.n_samples = \
            n_digits, depth, p_subop, max_length, split, n_samples
        self.n_nums = n_nums or self.N_NUMS
        self.maxval = int(self.n_nums ** self.n_digits)

        self.limits_per_depth = {i: int(min(n_possible(i), n_samples*2) * 0.75) for i in range(depth[0], depth[1]+1)}

        self.construct_vocab()
        self.data = self.load_cache()


    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "in": np.asarray(self.data[item].input, np.int8),
            "out": self.data[item].output,
            "in_len": len(self.data[item].input),
            "out_len": 1,
            "depth": self.data[item].depth
        }

    def __len__(self) -> int:
        return len(self.data)

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: self.out_vocabulary([x])[0], max_bad_samples=100)
