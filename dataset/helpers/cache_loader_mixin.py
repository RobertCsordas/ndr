import os
import torch
from typing import Any


class CacheLoaderMixin:
    VERSION = 0
    cache_dir = "./cache"

    def config_id(self) -> str:
        raise NotImplementedError()

    def generate_dataset(self) -> Any:
        raise NotImplementedError()

    def load_cache(self):
        fname = f"{self.cache_dir}/{self.__class__.__name__}/{self.config_id()}.pth"
        if os.path.isfile(fname):
            data = torch.load(fname)
            if data["version"] == self.VERSION:
                return data["data"]

        print("Generating dataset...")
        res = self.generate_dataset()
        print("Saving cache...")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        torch.save({"data": res, "version": self.VERSION}, fname)
        print("Done.")
        return res
