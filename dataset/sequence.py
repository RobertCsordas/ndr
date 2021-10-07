from dataclasses import replace
import framework
from models.encoder_decoder import add_eos
from typing import Tuple, Dict, Any, Optional, Callable, List, Type
import torch
import torch.nn.functional as F
import numpy as np


class AccuracyCounter:
    def __init__(self):
        self.n_total = 0
        self.n_ok = 0

    def get(self) -> float:
        return self.n_ok / self.n_total

    def add(self, ok_mask: torch.Tensor):
        self.n_total += ok_mask.nelement()
        self.n_ok += ok_mask.long().sum().item()


class TypedAccuracyCounter:
    def __init__(self, type_names: List[str]):
        self.counters = {}
        self.type_names = type_names

    def add(self, ok_mask: torch.Tensor, types: torch.Tensor):
        for t in torch.unique(types.int()).cpu().numpy().tolist():
            mask = types == t
            c = self.counters.get(t)
            if c is None:
                self.counters[t] = c = AccuracyCounter()

            c.add(ok_mask[mask])

    def get(self) -> Dict[str, Any]:
        return {f"accuracy/{self.type_names[t]}": data.get() for t, data in self.counters.items()}      


class SampleTracker:
    def __init__(self, header: List[str]):
        self.header = header
        self.enabled = [True for _ in header]
        self.n_cols = len(self.enabled)
        self.list = []

    def set_enabled(self, c: str, enabled: bool = True):
        c = self.header.index(c)
        if self.enabled[c] != enabled:
            assert not self.list
            self.enabled[c] = enabled
            self.n_cols = sum(int(a) for a in self.enabled)

    def add(self, *args):
        assert len(args) == self.n_cols
        self.list.append(args)

    def pick_seqeunces(self, limit: Optional[int]) -> List[List[str]]:
        if not limit or len(self.list) <= limit:
            return self.list
        s = np.random.RandomState(0).permutation(len(self.list))[:limit].tolist()
        return [self.list[i] for i in s]

    def get(self, limit: Optional[int] = None) -> framework.visualize.plot.TextTable:
        return framework.visualize.plot.TextTable([n for n, e in zip(self.header, self.enabled) if e],
                                                  self.pick_seqeunces(limit))


class SequenceTestState:
    def __init__(self, batch_dim: int = 1):
        self.cntr = AccuracyCounter()
        self.batch_dim = batch_dim
        self.time_dim = 1 - self.batch_dim

    def is_index_tensor(self, net_out: torch.Tensor) -> bool:
        return net_out.dtype in [torch.long, torch.int, torch.int8, torch.int16]

    def convert_to_index(self, net_out: torch.Tensor):
        return net_out if self.is_index_tensor(net_out) else net_out.argmax(-1)

    def compare_direct(self, net_out: Tuple[torch.Tensor, Optional[torch.Tensor]], ref: torch.Tensor,
                       ref_len: torch.Tensor):
        scores, len = net_out
        out = self.convert_to_index(scores)

        if len is not None:
            # Dynamic-length output
            if out.shape[0] > ref.shape[0]:
                out = out[: ref.shape[0]]
            elif out.shape[0] < ref.shape[0]:
                ref = ref[: out.shape[0]]

            unused = torch.arange(0, out.shape[0], dtype=torch.long, device=ref.device).unsqueeze(self.batch_dim) >= \
                     ref_len.unsqueeze(self.time_dim)

            ok_mask = ((out == ref) | unused).all(self.time_dim) & (len == ref_len)
        else:
            # Allow fixed lenght output
            assert out.shape==ref.shape
            ok_mask = (out == ref).all(self.time_dim)

        return ok_mask

    def compare_output(self, net_out: Tuple[torch.Tensor, Optional[torch.Tensor]], data: Dict[str, torch.Tensor]):
        return self.compare_direct(net_out, data["out"], data["out_len"])

    def step(self, net_out: Tuple[torch.Tensor, Optional[torch.Tensor]], data: Dict[str, torch.Tensor]):
        ok_mask = self.compare_output(net_out, data)
        self.cntr.add(ok_mask)

    @property
    def accuracy(self):
        return self.cntr.get()

    def plot(self) -> Dict[str, Any]:
        return {"accuracy/total": self.accuracy}


class SampleTrackerTestState(SequenceTestState):
    def __init__(self, batch_dim: int, max_bad_samples: int = 100, max_good_samples: int = 0, 
                 type_names: Optional[List[str]] = None):
        super().__init__(batch_dim=batch_dim)
        self.max_bad_samples = max_bad_samples
        self.max_good_samples = max_good_samples
        self.type_names = type_names

        self.bad_sequences = SampleTracker(["Input", "Reference", "Type", "Output"] + self.extra_ok_columns())
        self.bad_sequences.set_enabled("Type", type_names is not None)
        self.good_sequences = SampleTracker(["Input", "Reference", "Type"])
        self.good_sequences.set_enabled("Type", type_names is not None)

    def ref_to_text(self, data: Dict[str, torch.Tensor], i: int) -> Tuple[str, str]:
        raise NotImplementedError()

    def net_out_to_text(self, net_out: Any, data: Dict[str, torch.Tensor], i: int) -> str:
        raise NotImplementedError()

    def extra_ok_columns(self) -> List[str]:
        return []

    def get_gt_sample(self, data: Dict[str, torch.Tensor], i: int, *args, **kwargs) -> List[str]:
        s = list(self.ref_to_text(data, i))
        if self.type_names is not None:
            s.append(self.type_names[int(data["type"][i].item())])
        return s

    def get_failed_sample(self, net_out: Any, data: Dict[str, torch.Tensor], i: int, *args, **kwargs) -> List[str]:
        s = list(self.get_gt_sample(data, i))
        s.append(self.net_out_to_text(net_out, data, i))
        return s

    def track_samples(self, net_out: Any, ok_mask: torch.Tensor, data: Dict[str, Any], *args, **kwargs):
        if self.max_bad_samples > 0:
            for i in torch.nonzero(~ok_mask).squeeze(-1):
                self.bad_sequences.add(*self.get_failed_sample(net_out, data, i, *args, **kwargs))

        if self.max_good_samples > 0:
            for i in torch.nonzero(ok_mask).squeeze(-1):
                self.good_sequences.add(*self.get_gt_sample(data, i, *args, **kwargs))

    def plot(self) -> Dict[str, Any]:
        res = super().plot()
        if self.max_bad_samples > 0:
            res["mistake_examples"] = self.bad_sequences.get(self.max_bad_samples)
        if self.max_good_samples > 0:
            res["successfull_examples"] = self.good_sequences.get(self.max_bad_samples)
        return res


class TextClassifierTestState(SampleTrackerTestState):
    def __init__(self, input_to_text: Callable[[torch.Tensor], str],
                 output_to_text: Callable[[torch.Tensor], str], batch_dim: int = 1,
                 max_bad_samples: int = 100, type_names: Optional[List[str]] = None,
                 max_good_samples: int = 0):
        super().__init__(batch_dim, max_bad_samples, max_good_samples, type_names)
        self.in_to_text = input_to_text
        self.out_to_text = output_to_text
        self.losses = []
        self.oks = []
        self.type_names = type_names
        self.type_counters = TypedAccuracyCounter(type_names or [])

    def ref_to_text(self, data: Dict[str, torch.Tensor], i: int) -> Tuple[str, str]:
        t_ref = self.out_to_text(data["out"][i].item())
        t_in = self.in_to_text(data["in"].select(self.batch_dim, i)[: int(data["in_len"][i].item())].cpu().numpy().
                               tolist())
        return t_in, t_ref

    def net_out_to_text(self, net_out: Any, _, i: int) -> str:
        out = self.convert_to_index(net_out)
        return self.out_to_text(out[i].cpu().numpy().item())

    def step(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]):
        out = self.convert_to_index(net_out)
        ok_mask = out == data["out"]

        self.track_samples(net_out, ok_mask, data)
        self.oks.append(ok_mask.cpu())
        self.cntr.add(ok_mask)

        if self.type_names is not None:
            self.type_counters.add(ok_mask, data["type"])

    def get_sample_info(self) -> Tuple[List[float], List[bool]]:
        return torch.cat(self.losses, 0).numpy().tolist(), torch.cat(self.oks, 0).numpy().tolist()

    def plot(self) -> Dict[str, Any]:
        res = super().plot()
        res.update(self.type_counters.get())
        return res


class TextSequenceTestState(SampleTrackerTestState):
    def extra_ok_columns(self) -> List[str]:
        return ["Prefix match", "Oracle match"]

    def __init__(self, input_to_text: Callable[[torch.Tensor], torch.Tensor],
                 output_to_text: Callable[[torch.Tensor], torch.Tensor], batch_dim: int = 1,
                 max_bad_samples: int = 100, min_prefix_match_len: int = 1, eos_id: int = -1,
                 type_names: Optional[List[str]] = None, max_good_samples: int = 100):
        super().__init__(batch_dim, max_bad_samples, max_good_samples, type_names)
        self.bad_sequences.set_enabled("Oracle match", False)
        self.max_bad_samples = max_bad_samples
        self.in_to_text = input_to_text
        self.out_to_text = output_to_text
        self.n_prefix_ok = 0
        self.n_oracle_ok = 0
        self.oracle_available = False
        self.min_prefix_match_len = min_prefix_match_len
        self.eos_id = eos_id
        self.losses = []
        self.oks = []
        self.type_names = type_names
        self.type_counters = TypedAccuracyCounter(type_names or [])

    def set_eos_to_neginf(self, scores: torch.Tensor) -> torch.Tensor:
        id = self.eos_id if self.eos_id >= 0 else (scores.shape[-1] + self.eos_id)
        return scores.index_fill(-1, torch.tensor([id], device=scores.device), float("-inf"))

    def loss(self, net_out: torch.Tensor, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        mask = torch.arange(net_out.shape[1-self.batch_dim], device=net_out.device).unsqueeze(1) <= \
               data["out_len"].unsqueeze(0)

        ref = add_eos(data["out"], data["out_len"], net_out.shape[-1] - 1)
        l = F.cross_entropy(net_out.flatten(end_dim=-2), ref.long().flatten(), reduction='none')
        l = l.reshape_as(ref) * mask
        nonbatchdims = tuple(i for i in range(l.ndim) if i!=self.batch_dim)
        l = l.sum(dim=nonbatchdims) / mask.sum(dim=nonbatchdims).float()
        return l

    def ref_to_text(self, data: Dict[str, torch.Tensor], i: int) -> Tuple[str, str]:
        t_ref = self.out_to_text(data["out"].select(self.batch_dim, i)[: int(data["out_len"][i].item())].
                cpu().numpy().tolist())
        t_in = self.in_to_text(data["in"].select(self.batch_dim, i)[: int(data["in_len"][i].item())].cpu().numpy().
                               tolist())
        return t_in, t_ref

    def net_out_to_text(self, net_out: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]], _, i: int) -> str:
        scores, out_len = net_out
        out = self.convert_to_index(scores)
        out_end = None if out_len is None else out_len[i].item()
        return self.out_to_text(out.select(self.batch_dim, i)[:out_end].cpu().numpy().tolist())

    def get_failed_sample(self, net_out: Any, data: Dict[str, torch.Tensor], i: int, prefix_ok: torch.Tensor, 
                          oracle_ok: torch.Tensor) -> List[str]:
        res = super().get_failed_sample(net_out, data, i)
        res.append(str(prefix_ok[i].item()))
        if self.oracle_available:
            res.append(str(oracle_ok[i].item()))
        return res

    def step(self, net_out: Tuple[torch.Tensor, Optional[torch.Tensor]], data: Dict[str, torch.Tensor]):
        ok_mask = self.compare_output(net_out, data)
        scores, _ = net_out

        if not self.is_index_tensor(scores):
            self.oracle_available = True
            self.bad_sequences.set_enabled("Oracle match", True)
            out_noeos = self.set_eos_to_neginf(scores).argmax(-1)
            oracle_ok = self.compare_direct((out_noeos, data["out_len"].clamp_(max=out_noeos.shape[1-self.batch_dim])),
                                             data["out"], data["out_len"])
            self.n_oracle_ok += oracle_ok.long().sum().item()

            self.losses.append(self.loss(net_out[0], data).cpu())
        else:
            oracle_ok = None

        prefix_len = data["out_len"] if net_out[1] is None else torch.minimum(data["out_len"], net_out[1])
        prefix_len = torch.minimum(prefix_len.clamp(min=self.min_prefix_match_len), data["out_len"])
        prefix_ok_mask = self.compare_direct((net_out[0], prefix_len), data["out"], prefix_len)

        self.track_samples(net_out, ok_mask, data, prefix_ok_mask, oracle_ok)

        self.oks.append(ok_mask.cpu())
        self.cntr.add(ok_mask)
        self.n_prefix_ok += prefix_ok_mask.long().sum().item()
        if self.type_names is not None:
            self.type_counters.add(ok_mask, data["type"])

    def get_sample_info(self) -> Tuple[List[float], List[bool]]:
        return torch.cat(self.losses, 0).numpy().tolist(), torch.cat(self.oks, 0).numpy().tolist()

    def plot(self) -> Dict[str, Any]:
        res = super().plot()
        res["accuracy/prefix"] = self.n_prefix_ok / self.cntr.n_total

        if self.oracle_available:
            res["accuracy/oracle"] = self.n_oracle_ok / self.cntr.n_total

        if self.losses:
            res["loss_histogram"] = framework.visualize.plot.Histogram(torch.cat(self.losses, 0))

        res.update(self.type_counters.get())
        return res
