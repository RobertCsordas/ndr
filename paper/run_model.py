import sys
import os

sys.path.insert(0, os.path.dirname(__file__)+"/../..")

import wandb
import torch
from main import initialize

class LoadedModel:
    def __init__(self, run: str, step: int, patch_args=None) -> None:
        api = wandb.Api()
        self.wrun = api.run(run)
        self.step = step
        self.patch_args = patch_args
        self.init_model()
        self.plots = {}

    def load_file(self, rel_path, load=True):
        rel_fname = f"activations/{self.wrun.id}/{rel_path}"
        if not os.path.isfile(rel_fname):
            self.wrun.file(rel_path).download(root=f"activations/{self.wrun.id}", replace=True)
        return torch.load(rel_fname) if load else rel_fname

    def log_hook(self, plotted):
        self.plots.update(plotted)
        self.log_orig(plotted)

    def init_model(self):
        ckpt = self.load_file(f"checkpoint/model-{self.step}.pth", load=False)
        data = torch.load(ckpt)

        if "wandb" in data["run_invariants"]["args"]["log"]:
            data["run_invariants"]["args"]["log"] = "tb"
            data["run_invariants"]["args"]["name"] = "tmp"
            data["run_invariants"]["args"]["reset"] = True

            if self.patch_args:
                self.patch_args(data["run_invariants"]["args"])

            torch.save(data, ckpt)

        self.helper, self.task = initialize(ckpt)
        self.log_orig = self.helper.log
        self.helper.log = self.log_hook
        self.helper.args.debug_plot_interval = 1

    def create_sample(self, inp, target):
        res = {
            "in": torch.tensor(self.task.train_set.in_vocabulary(inp), dtype=torch.uint8).unsqueeze(1),
            "in_len": torch.tensor([len(inp)], dtype=torch.uint8),
            "out": torch.tensor(self.task.train_set.out_vocabulary(target), dtype=torch.uint8),
            "out_len": torch.tensor([len(target)], dtype=torch.uint8)
        }
        res = self.helper.to_device(res)
        return res

    def run(self, sample, target_out):
        sample = self.create_sample(sample, target_out)
        self.task.validation_started_on = "hack"
        self.task.model.eval()
        with torch.no_grad():
            print(self.task.run_model(sample))
