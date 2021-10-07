import torch
import torch.nn
import torch.optim
import framework
import torch.utils.data
import torch.cuda.amp
from typing import Optional, Dict, Any, Tuple, List
from interfaces import Result
from ..task import Task
from layers import LayerRegularizer
import time


class SimpleTask(Task):
    MAX_LENGHT_PER_BATCH = None
    train_set: torch.utils.data.Dataset
    train_loader: torch.utils.data.DataLoader
    model: torch.nn.Module

    def create_datasets(self):
        raise NotImplementedError()

    def create_model_interface(self):
        raise NotImplementedError()

    def create_model(self) -> torch.nn.Module:
        raise NotImplementedError()

    def create_state(self):
        pass

    @property
    def amp_enabled(self):
        return torch.cuda.is_available() and self.helper.args.amp

    @property
    def time_dim(self) -> int:
        return 1 - self.batch_dim

    def __init__(self, helper: framework.helpers.TrainingHelper):
        super().__init__(helper)

        self.avg_num_chunks = framework.utils.Average()
        self.max_grad = 0
        self.time_sum = 0

        self.create_datasets()
        self.create_loaders()
        self.model = self.create_model()
        self.model = self.model.to(self.helper.device)

        self.create_model_interface()
        self.create_optimizer()

        self.regularizer = LayerRegularizer(self.model)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)
        self.helper.saver["scaler"] = self.scaler

        print(f"Total number of model parameters: {sum(p.numel() for p in self.model.parameters())}")

        self.helper.saver["model"] = self.model
        self.create_state()
        self.helper.restore()

        self.fetcher = None

    def fetch_thread(self):
        data = self.prepare_data(self.get_train_batch())
        n_chunks = self.get_n_chunks(data)
        d_chunks = self.chunk_batch_dim(data, n_chunks)

        return data, d_chunks

    def create_train_loader(self, loader: torch.utils.data.Dataset, seed: Optional[int] = None) -> \
            torch.utils.data.DataLoader:

        return super().create_train_loader_bs(loader, self.helper.args.batch_size, seed)

    def create_loaders(self):
        self.train_loader = self.create_train_loader(self.train_set)
        self.valid_loaders = framework.data_structures.DotDict()
        self.valid_loaders.update({k: self.create_valid_loader(v) for k, v in self.valid_sets.items()})

    def create_optimizer(self):
        if self.helper.args.optimizer in ["adam", "adamw"]:
            opt = torch.optim.Adam if self.helper.args.optimizer == "adam" else torch.optim.AdamW
            self.set_optimizer(opt(self.model.parameters(), self.helper.args.lr,
                                                weight_decay=self.helper.args.wd, betas=self.helper.args.adam.betas,
                                                eps=self.helper.args.adam.eps))
        elif self.helper.args.optimizer == "sgd":
            self.set_optimizer(torch.optim.SGD(self.model.parameters(), self.helper.args.lr,
                                               weight_decay=self.helper.args.wd, momentum=0.9))
        else:
            assert False, f"Unsupported optimizer: {self.helper.args.optimizer}"

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.helper.saver.register("optimizer", self.optimizer, replace=True)

    def get_train_batch(self) -> Dict[str, Any]:
        return next(self.data_iter)

    def chunk_batch_dim(self, data: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
        if n == 1:
            return [data]

        res = [{} for _ in range(n)]
        for k, v in data.items():
            assert torch.is_tensor(v), "Only tensors are supported by autosplitting"

            bd = self.batch_dim if self.batch_dim < v.ndimension() else 0
            assert v.shape[bd] % n == 0, f"Batch (dim {bd} of input {k} of shape {v.shape} is not divisible by {n})"

            for i, c in enumerate(v.chunk(n, dim=bd)):
                res[i][k] = c

        return res

    def is_seq2seq_task(self, data: Dict[str, Any]) -> bool:
        return "in_len" in data and "out_len" in data

    def get_seq_length(self, data: Dict[str, Any]) -> int:
        # This assumes separate encoder and decoder
        return max(data["in"].shape[self.time_dim], data["out"].shape[self.time_dim] if data["out"].ndim > 1 else 0)

    def get_n_chunks(self, data: Dict[str, Any]) -> int:
        max_length_per_batch = self.helper.args.max_length_per_batch or self.MAX_LENGHT_PER_BATCH
        if self.is_seq2seq_task(data) and max_length_per_batch:
            # The formula below assumes quadratic memory consumption
            return int(2**int(self.get_seq_length(data) / max_length_per_batch))
        return 1

    def post_backward(self) -> Dict[str, Any]:
        return {}

    def train_step(self) -> Tuple[Result, Dict[str, Any]]:
        plots = {}

        with self.forward_time_meter:
            self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)

            data, d_chunks = self.fetcher.get()

            res_list = []
            weights = []

            self.avg_num_chunks.add(len(d_chunks))

            total_out_len = data["out_len"].sum() if "out_len" in data else 1

            if self.helper.args.speedtest:
                torch.cuda.synchronize()
                start_time = time.time()

            for d in d_chunks:
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    res, custom_plots = self.run_model(d)
                    res_list.append(res)
                    plots.update(custom_plots)
                weights.append((d["out_len"].sum()/total_out_len) if "out_len" in d else 1)
                total_loss = res_list[-1].loss + self.regularizer.get()
                assert torch.isfinite(total_loss)
                self.scaler.scale(total_loss * weights[-1]).backward()
                plots.update(self.post_backward())

            self.scaler.unscale_(self.optimizer)
            if self.helper.args.grad_clip:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.helper.args.grad_clip)
                self.max_grad = max(self.max_grad, gn)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.helper.args.speedtest:
                torch.cuda.synchronize()
                end_time = time.time()

            self.helper.state.iter += 1
            res = res_list[0].__class__.merge(res_list, weights)

            # if self.helper.state.iter % 20 == 0:

        if "in_len" in data and self.helper.args.speedtest:
            self.time_sum += (end_time - start_time) / data["in_len"].max()

        if self.helper.state.iter % 20 == 0:
            plots["timing/ms_per_token"] = (self.time_sum / 20) * 1000
            self.time_sum = 0

        return res, plots

    def plot(self, res: Result) -> Dict[str, Any]:
        res = super().plot(res)

        if self.helper.state.iter % 20 == 0:
            res["average_num_chunks"] = self.avg_num_chunks.get()
            if self.helper.args.grad_clip:
                res["max_grad"] = self.max_grad
                self.max_grad = 0

        return res

    def train(self):
        self.loss_average.reset()

        self.data_iter = iter(self.train_loader)
        self.fetcher = framework.helpers.StoppingParallelProducer(self.fetch_thread)

        try:
            while (self.helper.args.stop_after or 10e10) > self.helper.state.iter:
                self.load_time_meter.stop()

                res, plots = self.train_step()
                plots.update(self.plot(res))

                with self.plot_time_meter:
                    self.helper.log(plots)

                self.load_time_meter.start()

                self.helper.tick()
        except self.fetcher.Stopped:
            pass
