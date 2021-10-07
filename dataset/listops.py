from framework.data_structures.vocabulary import WordVocabulary
from genericpath import exists
import numpy as np
from numpy.random import randint, sample
from typing import Callable, Tuple, Union, List, Optional, Dict, Any
import dataclasses
import os
import framework
import torch
import torch.utils.data
from tqdm import tqdm
from .sequence import TextClassifierTestState
import math
import multiprocessing
import hashlib


RangeOrInt = Union[Tuple[int, int], int]


@dataclasses.dataclass
class Sample:
    input: Union[str, List[int]]
    output: int
    depth: int
    length: int
    max_dependency_depth: int

    def dup(self, **overwrite_args):
        d = dataclasses.asdict(self)
        d.update(overwrite_args)
        return self.__class__(**d)


def get_selector_op(selector: Callable[[List[Sample]], int]) -> Callable[[List[Sample]], Sample]:
    def op(x):
        y = selector(x)
        return x[y].dup(max_dependency_depth=x[y].max_dependency_depth + 1, depth=max(a.depth for a in x) + 1)
    return op


def sum_op(x: List[Sample]) -> Sample:
    return Sample("", sum(a.output for a in x) % 10, max(a.depth for a in x) + 1, -1,
                  max(a.max_dependency_depth for a in x) + 1)


def median_op(x: List[Sample]) -> Sample:
    # Median op compatible with original listops
    indices = np.argsort([a.output for a in x])
    selected = [indices[len(x) // 2]] if len(x) % 2 == 1 else [indices[len(x) // 2 - 1], indices[len(x) // 2]]
    res = int(sum(x[i].output for i in selected) / len(selected))

    return Sample("", res, max(a.depth for a in x) + 1, -1, max(x[i].max_dependency_depth for i in selected) + 1)


class ListopsTestState(TextClassifierTestState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hists = {
            "ok_dependency_depts": [],
            "not_ok_dependency_depts": []
        }

        self.counts_per_depth = {}

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        super().step(net_out, data)

        out = self.convert_to_index(net_out)
        ok_mask = out == data["out"]

        self.hists["ok_dependency_depts"] += data["max_dependency_depth"][ok_mask].cpu().numpy().tolist()
        self.hists["not_ok_dependency_depts"] += data["max_dependency_depth"][~ok_mask].cpu().numpy().tolist()

        depths = torch.unique(data["max_dependency_depth"]).int().cpu().numpy().tolist()
        for d in depths:
            m = data["max_dependency_depth"] == d
            if d not in self.counts_per_depth:
                self.counts_per_depth[d] = {"ok": 0, "total": 0}

            self.counts_per_depth[d]["total"] += m.long().sum().item()
            self.counts_per_depth[d]["ok"] += ok_mask[m].long().sum().item()


    def plot(self) -> Dict[str, Any]:
        res = super().plot()
        for name, h in self.hists.items():
            res[f"hist/{name}"] = framework.visualize.plot.Histogram(np.asarray(h, dtype=np.int32))

        for k, v in self.counts_per_depth.items():
            res[f"dep_length/{k}/accuracy"] = v["ok"] / v["total"]

        return res


class ListOps(torch.utils.data.Dataset):
    VERSION = 7
    in_vocabulary: Optional[framework.data_structures.WordVocabulary] = None
    out_vocabulary: Optional[framework.data_structures.WordVocabulary] = None

    OPS = [
        ("MIN", get_selector_op(lambda x: min(range(len(x)), key=lambda i: x[i].output))),
        ("MAX", get_selector_op(lambda x: max(range(len(x)), key=lambda i: x[i].output))),
        ("MED", median_op),
        # ("MED", get_selector_op(lambda x: np.argsort([a.output for a in x])[len(x) // 2])),
        ("SM", sum_op)
    ]

    def get_one(self, seed) -> Sample:
        def sample_subop(args: List[Sample]) -> Sample:
            op = self.OPS[seed.randint(len(self.OPS))]
            res = op[1](args)
            res.input = f"[{op[0]} {' '.join(a.input for a in args)} ]"
            res.length = 3 + sum(a.length for a in args)
            return res

        def gen_args(max_args: int, max_depth: int) -> List[Sample]:
            n_args = seed.randint(2, max_args + 1) if (max_args+1) != 2 else 2
            return [get_subtree(max_depth) for _ in range(n_args)]

        def get_subtree(max_depth: int) -> Sample:
            if (max_depth == 1) or (seed.rand() > self.p_op):
                val = seed.randint(10)
                return Sample(str(val), val, 1, 1, 1)
            else:
                return sample_subop(gen_args(self.max_args, max_depth - 1))

        res = get_subtree(self.depth[1])

        # Add ops until it reaches the minimal depth
        while res.depth < self.depth[0]:
            args = gen_args(self.max_args - 1, self.depth[1] - 1)
            # Insert the old subtree: guaranteed to grow by 1
            args.insert(seed.randint(len(args)), res)   
            res = sample_subop(args)

        return res

    def config_id(self) -> str:
        res = f"{self.length}_{self.depth}_{self.p_op}_{self.max_args}_{self.set}_{self.eq_depth}_{self.n_samples}"
        if self.custom_vocab:
            res = res + "_voc" + hashlib.md5((str(self.in_vocabulary) + str(self.out_vocabulary)).encode()).hexdigest()
        return res

    def translate_sample(self, s: Sample) -> Sample:
        return s.dup(input=self.in_vocabulary(s.input), output=self.out_vocabulary([str(s.output)])[0])

    def load_cache(self):
        fname = f"{self.cache_dir}/{self.__class__.__name__}/{self.config_id()}.pth"
        if os.path.isfile(fname):
            data = torch.load(fname)
            if data["version"] == self.VERSION:
                self.data = data["data"]
                return

        print("Generating dataset...")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        seed = np.random.RandomState(0x12345678+hash(self.set) & 0x1FFFFFFF)
        self.data = []

        def get_samples(seed):
            seed = np.random.RandomState(seed)
            res = []
            for _ in range(100):
                s = self.get_one(seed)
                if not (self.length[0] <= s.length <= self.length[1]):
                    continue
                res.append(s)
            return res

        nproc = multiprocessing.cpu_count()
        pbar = tqdm(total = self.n_samples)

        lim = math.ceil(self.n_samples / (self.depth[1] - self.depth[0] + 1))
        depth_bins = {r: 0 for r in range(self.depth[0], self.depth[1] + 1)}

        seed_start = seed.randint(0x000FFFFF)
        seed_offset = 0

        known = set()
        with framework.utils.ParallelMapPool(get_samples) as ppool:
            while len(self.data) < self.n_samples:
                seeds = [seed_start + seed_offset * nproc + i for i in range(nproc)]
                seed_offset += nproc

                # Generate many
                res = ppool.map(seeds)
                res = sum(res, [])

                for s in res:
                    h = hash(s.input)

                    # Ensure there are no repeats
                    if h in known:
                        continue
                    known.add(h)

                    # Rejection sample based on length
                    if self.eq_depth:
                        if (s.max_dependency_depth not in depth_bins) or depth_bins[s.max_dependency_depth] >= lim:
                            continue
                        depth_bins[s.max_dependency_depth] += 1

                    self.data.append(self.translate_sample(s))
                    pbar.update(1)

                    if len(self.data) >= self.n_samples:
                        break

        torch.save({"data": self.data, "version": self.VERSION}, fname)
        print("Done.")

    def construct_vocab(self):
        if self.in_vocabulary is None:
            digits = [str(i) for i in range(10)]
            ListOps.in_vocabulary = framework.data_structures.WordVocabulary(["]"] + ["["+o[0] for o in self.OPS] + digits, 
                                                                             split_punctuation=False)
            ListOps.out_vocabulary = framework.data_structures.WordVocabulary(digits)

    def print_hist(self):
        h = {}
        res_h = {}
        for s in self.data:
            h[s.max_dependency_depth] = h.get(s.max_dependency_depth, 0) + 1
            res_h[s.output] = res_h.get(s.output, 0) + 1
        ks = list(sorted(h.keys()))
        cs = np.cumsum([h[k] for k in ks])
        print("Listops dependency")
        print("    depth histogram: ", ", ".join([f"{k}: {h[k]}" for k in ks]))
        print("    depth cumulative histogram: ", ", ".join([f"{k}: {cs[i]/cs[-1]*100:.1f}%" for i, k in enumerate(ks)]))

        o_ks = list(sorted(res_h.keys()))
        print("    out histogram: ", ", ".join([f"{k}: {res_h[k]/len(self)*100:.1f}%" for k in o_ks]))

    def __init__(self, set: str, length: RangeOrInt, depth: RangeOrInt = 20, p_op: float = 0.25, max_args: int = 5,
                 n_samples: int = 10000, cache_dir: str = "./cache", equivalize_depdendency_depth: bool = False,
                 in_vocab: Optional[WordVocabulary] = None, out_vocab: Optional[WordVocabulary] = None) -> None:
        self.length = (0, length) if isinstance(length, int) else length
        self.depth = (2, depth) if isinstance(depth, int) else depth
        self.p_op = p_op
        self.max_args = max_args
        self.n_samples = n_samples
        self.cache_dir = cache_dir
        self.set = set
        self.eq_depth = equivalize_depdendency_depth
        self.custom_vocab = in_vocab is not None
        assert (in_vocab is None) == (out_vocab is None)
        if self.custom_vocab:
            self.in_vocabulary = in_vocab
            self.out_vocabulary = out_vocab
        else:
            self.construct_vocab()

        with framework.utils.LockFile(os.path.join(self.cache_dir, "lock")):
            self.load_cache()

        self.print_hist()

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return {
            "in": np.asarray(self.data[item].input, np.int8),
            "out": self.data[item].output,
            "in_len": len(self.data[item].input),
            "out_len": 1,
            "max_dependency_depth": self.data[item].max_dependency_depth
        }

    def __len__(self) -> int:
        return len(self.data)

    def start_test(self) -> TextClassifierTestState:
        return ListopsTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                lambda x: self.out_vocabulary([x])[0], max_bad_samples=1000)
