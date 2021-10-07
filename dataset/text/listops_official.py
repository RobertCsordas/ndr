import os

from framework.utils import download
import framework
from .text_dataset import TextDataset, TextDatasetCache
import re
from ..sequence import TextClassifierTestState
from typing import Any, Dict
from .listops_official_generator import generate_tree, to_string, to_value
from tqdm import tqdm
from ..helpers.rejection_sampler import rejection_sample
import numpy as np


class ListopsOfficial(TextDataset):
    URLS = {
        "train": "https://raw.githubusercontent.com/nyu-mll/spinn/listops-release/python/spinn/data/listops/train_d20s.tsv",
        "test": "https://raw.githubusercontent.com/nyu-mll/spinn/listops-release/python/spinn/data/listops/test_d20s.tsv",
    }

    def build_cache(self) -> TextDatasetCache:
        index_table = {"simple": {}}
        in_sentences = []
        out_sentences = []

        for set in ["test", "train"]:
            url = self.URLS[set]
            set_fn = os.path.join(self.cache_dir, os.path.split(url)[-1])
            os.makedirs(os.path.dirname(set_fn), exist_ok=True)

            print("Downloading", url)
            download(url, f"{set_fn}", ignore_if_exists=True)

            this_set = []
            index_table["simple"][set] = this_set

            with open(set_fn) as f:
                for line in f:
                    outp, inp = line.split("\t")
                    in_sentences.append(re.sub("((\\(|\\)) *)+", "", inp).strip())
                    out_sentences.append(outp.strip())
                    this_set.append(len(in_sentences) - 1)

        return TextDatasetCache().build(index_table, in_sentences, out_sentences, split_punctuation=False)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        res = super().__getitem__(item)
        res["out"] = res["out"][0]
        return res

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: self.out_vocabulary([x])[0], max_bad_samples=1000)


class ListopsOfficialGenerate(TextDataset):
    def get_cache_fname(self) -> str:
        return f"cache_{self.N}.pth"

    def build_cache(self) -> TextDatasetCache:
        index_table = {"simple": {}}
        in_sentences = []
        out_sentences = []

        this_set = []
        index_table["simple"]["train"] = this_set

        print(f"Generating ListOps with {self.N} elements")
        samples = rejection_sample(self.N, np.random.RandomState(), lambda x: generate_tree(1), lambda x: True, hash)

        print("Postprocessing...")

        def process(example):
            outp = str(to_value(example))
            inp = to_string(example)
            return re.sub("((\\(|\\)) *)+", "", inp).strip(), outp.strip()

        with framework.utils.ParallelMapPool(process) as map:
            samples = map.map(samples)

        for i, o in tqdm(samples):
            in_sentences.append(i)
            out_sentences.append(o)
            this_set.append(len(in_sentences) - 1)

        return TextDatasetCache().build(index_table, in_sentences, out_sentences, split_punctuation=False)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        res = super().__getitem__(item)
        res["out"] = res["out"][0]
        return res

    def __init__(self, N: int, shared_vocabulary: bool = False, cache_dir: str="./cache/"):
        self.N = N
        super().__init__(sets=["train"], split_type=["simple"], cache_dir=cache_dir, shared_vocabulary=shared_vocabulary)

    def start_test(self) -> TextClassifierTestState:
        return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                       lambda x: self.out_vocabulary([x])[0], max_bad_samples=1000)
