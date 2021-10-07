import framework
from interfaces import Result, ModelInterface
import torch
import torch.utils.data
from tqdm import tqdm
from typing import Dict, Any, Iterable, Tuple, Optional
import os
import optimizer
from dataclasses import dataclass

@dataclass
class LastBestMarker:
    iter: int
    loss: float
    accuracy: float


class Task:
    valid_loaders: framework.data_structures.DotDict
    model_interface: ModelInterface
    batch_dim: int
    TRAIN_NUM_WORKERS = 1
    VALID_NUM_WORKERS = 1

    def __init__(self, helper: framework.helpers.TrainingHelper):
        self.helper = helper
        self.helper.state.best_losses = {}
        self.helper.state.best_accuracies = {}
        self.valid_sets = framework.data_structures.DotDict()
        self.loss_average = framework.utils.Average()
        self.forward_time_meter = framework.utils.ElapsedTimeMeter()
        self.load_time_meter = framework.utils.ElapsedTimeMeter()
        self.plot_time_meter = framework.utils.ElapsedTimeMeter()

        if self.helper.args.lr_sched.type == "step":
            self.lr_scheduler = optimizer.StepLrSched(self.helper.args.lr, self.helper.args.lr_sched.steps,
                                                      self.helper.args.lr_sched.gamma)

        elif self.helper.args.lr_sched.type == "noam":
            self.lr_scheduler = optimizer.NoamLRSched(self.helper.args.lr, self.helper.args.state_size,
                                                      self.helper.args.lr_warmup)
        else:
            assert False

    def create_valid_loader(self, vset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
        batch_size = self.test_batch_size

        # Do bucketed testing even when the bucketed training is not enabled
        if "in_len" in vset[0]:
            batch_sampler = framework.loader.sampler.BucketedSampler(vset, batch_size, infinite=False, long_first=True,
                                                                     random_order=False)
            batch_size = 1
        else:
            batch_sampler = None

        return torch.utils.data.DataLoader(vset, batch_size=batch_size, batch_sampler=batch_sampler,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)


    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set, mask = False)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def replace_valid_set(self, name: str, vset: torch.utils.data.Dataset):
        self.valid_sets[name] = vset
        self.valid_loaders[name] = self.create_valid_loader(vset)

    def create_train_loader_bs(self, loader: torch.utils.data.Dataset, batch_size: int, seed: Optional[int] = None) \
                            -> torch.utils.data.DataLoader:

        if self.helper.args.length_bucketed_sampling and "in_len" in loader[0]:
            batch_sampler = framework.loader.sampler.BucketedSampler(loader, batch_size, infinite=True, drop_last=True,
                                                                     random_order=True)
            sampler = None
            batch_size = 1
        else:
            batch_sampler = None
            sampler = framework.loader.sampler.InfiniteSampler(loader, seed = seed)


        return torch.utils.data.DataLoader(loader, batch_size=batch_size,
                                           sampler=sampler, batch_sampler=batch_sampler,
                                           collate_fn=framework.loader.collate.VarLengthCollate(
                                               batch_dim=self.batch_dim),
                                           num_workers=self.TRAIN_NUM_WORKERS, pin_memory=True,
                                           persistent_workers=self.TRAIN_NUM_WORKERS > 0)

    def create_validate_on_train(self, set: torch.utils.data.Dataset):
        self.valid_sets.train = set
        self.valid_loaders.train = torch.utils.data.DataLoader(set, batch_size=self.helper.args.batch_size,
                                   collate_fn=framework.loader.collate.VarLengthCollate(batch_dim=self.batch_dim),
                                   sampler=framework.loader.sampler.SubsetSampler(set, len(self.valid_sets.iid)
                                                                          if "iid" in self.valid_sets else 1000),
                                   num_workers=self.VALID_NUM_WORKERS, persistent_workers=self.VALID_NUM_WORKERS > 0)

    def clip_gradients(self):
        if self.helper.args.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)

    def set_optimizer_lr(self, lr: float):
        framework.utils.set_lr(self.optimizer, lr)
        if self.helper.state.iter % 100 == 0:
            self.helper.log({"lr": lr})

    def set_linear_warmup(self, curr_step: int, n_steps: int, final: float) -> float:
        if curr_step >= n_steps:
            lr = final
        else:
            lr = final / n_steps * (curr_step+1)

        self.set_optimizer_lr(lr)
        return lr

    def set_lr(self):
        if self.helper.args.lr_sched.type == "step":
            self.set_linear_warmup(self.helper.state.iter, self.helper.args.lr_warmup,
                                   self.lr_scheduler.get(self.helper.state.iter))
        elif self.helper.args.lr_sched.type == "noam":
            self.set_optimizer_lr(self.lr_scheduler.get(self.helper.state.iter))
        else:
            assert False

    def prepare_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.helper.to_device(data)

    def run_model(self, data: Dict[str, torch.Tensor]) -> Tuple[Result, Dict[str, Any]]:
        res = self.model_interface(data)
        return res, {}

    def validate_on(self, set: torch.utils.data.Dataset, loader: torch.utils.data.DataLoader) -> Tuple[Any, float]:
        self.model.eval()

        with torch.no_grad():
            loss_sum = 0

            test = set.start_test()
            for d in tqdm(loader):
                d = self.prepare_data(d)
                res, _ = self.run_model(d)
                digits = self.model_interface.decode_outputs(res)
                loss_sum += res.loss.sum().item() * res.batch_size

                test.step(digits, d)

        self.model.train()
        return test, loss_sum / len(set)

    def validate_on_name(self, name: str) -> Tuple[Any, float]:
        return self.validate_on(self.valid_sets[name], self.valid_loaders[name])

    def update_best_accuracies(self, name: str, accuracy: float, loss: float):
        if name not in self.helper.state.best_losses or loss < self.helper.state.best_losses[name].loss:
                self.helper.state.best_losses[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        if name not in self.helper.state.best_accuracies or accuracy > \
                self.helper.state.best_accuracies[name].accuracy:
            self.helper.state.best_accuracies[name] = LastBestMarker(self.helper.state.iter, loss, accuracy)

        return {
            f"{name}/time_since_best_loss": self.helper.state.iter - self.helper.state.best_losses[name].iter,
            f"{name}/time_since_best_accuracy": self.helper.state.iter - self.helper.state.best_accuracies[name].iter
        }

    def validate_on_names(self, name_it: Iterable[str]) -> Dict[str, Any]:
        charts = {}
        sum_accuracy = 0
        sum_all_losses = 0

        for name in name_it:
            test, loss = self.validate_on_name(name)

            print(f"Validation accuracy on {name}: {test.accuracy}")
            charts[f"{name}/loss"] = loss
            sum_all_losses += loss
            charts.update({f"{name}/{k}": v for k, v in test.plot().items()})
            sum_accuracy += test.accuracy
            
            charts.update(self.update_best_accuracies(name, test.accuracy, loss))

        charts["mean_accuracy"] = sum_accuracy / len(self.valid_sets)
        charts["mean_loss"] = sum_all_losses / len(self.valid_sets)
        return charts

    def validate(self) -> Dict[str, Any]:
        return self.validate_on_names(self.valid_sets.keys())

    def plot(self, res: Result) -> Dict[str, Any]:
        plots = {}

        self.loss_average.add(res.loss)

        if self.helper.state.iter % 200 == 0:
            plots.update(res.plot())

        if self.helper.state.iter % 20 == 0:
            plots["train/loss"] = self.loss_average.get()
            plots["timing/ms_per_iter"] = self.forward_time_meter.get(True)*1000/20
            plots["timing/ms_per_load"] = self.load_time_meter.get(True)*1000/20
            plots["timing/ms_per_plot"] = self.plot_time_meter.get(True)*1000/20

        if self.helper.state.iter % self.helper.args.test_interval == 0:
            plots.update({f"validation/{k}": v for k, v in self.validate().items()})

        return plots

    def train_step_reconfig(self):
        pass

    def create_model_interface(self):
        raise NotImplementedError()

    def create_datasets(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def save_weights(self):
        pass

    def load_weights(self, file_path: str):
        pass

    def post_train(self):
        pass

    def finish(self):
        pass

    @property
    def test_batch_size(self):
        return self.helper.args.test_batch_size or self.helper.args.batch_size
