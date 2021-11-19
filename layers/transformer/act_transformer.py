import torch
import torch.nn
from .transformer import TransformerEncoderLayer, AttentionMask
from ..regularized_layer import RegularizedLayer
from typing import Optional


class ACTTransformerEncoder(RegularizedLayer, torch.nn.Module):
    def __init__(self, layer, n_layers: int, d_model: int, threshold: float, act_loss_weight: float, 
                 ut_variant: bool = False, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, d_model=d_model, **kwargs)
        self.set_n_layers(n_layers)
        self.p_map = torch.nn.Linear(d_model, 1)
        self.threshold = threshold
        self.act_loss_weight = act_loss_weight
        self.len_stats = {}
        self.ut_variant = ut_variant

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def add_different_shape(self, a, b) -> torch.Tensor:
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            return a + b
        assert a.ndim == 1 and b.ndim == 1
        if a.shape[0] < b.shape[0]:
            a, b = b, a

        res = a.clone()
        res[:b.shape[0]] += b
        return res

    def update_len_stat(self, n_updates: torch.Tensor, lens: torch.Tensor):
        for l in lens.unique():
            l = l.item()
            s = self.len_stats.get(l)
            if s is None:
                s = {"sum": 0, "val": 0}
                self.len_stats[l] = s

            mask = lens == l
            s["sum"] += mask.int().sum().item()
            s["val"] += n_updates[mask].int().sum().item()

    def forward(self, data: torch.Tensor, mask: Optional[AttentionMask] = None, *args, **kwargs):
        # Data shape: [n_batch, n_steps, data_size]
        bs, n, _ = data.shape

        p_halt = torch.zeros((bs, n, 1), device=data.device, dtype=torch.float32) 
        has_mask = mask is not None and mask.src_length_mask is not None
        if has_mask:
            # Masked columns are already "stopped"
            p_halt += mask.src_length_mask.unsqueeze(-1).float()

        remainders = torch.zeros_like(p_halt)
        n_updates = torch.zeros_like(p_halt)
        out_data = 0
        for i, l in enumerate(self.layers):
            # Based on https://arxiv.org/pdf/1807.03819.pdf, page 14
            p = torch.sigmoid(self.p_map(data))

            still_running = (p_halt < 1.0).float()

            stopped = ((p_halt + p * still_running) > self.threshold).float()
            if (i == len(self.layers) - 1) and (not self.ut_variant):
                # If it is the last layer, everyone is stopped. Eq (6), https://arxiv.org/pdf/1603.08983.pdf.
                # https://arxiv.org/pdf/1807.03819.pdf does not handle this properly.
                stopped.fill_(1.0)

            new_halted = stopped * still_running
            still_running = (1 - stopped) * still_running

            p_halt += p * still_running
            remainders += new_halted * (1 - p_halt)
            p_halt += new_halted * remainders

            n_updates += still_running + new_halted
            update_weights = p * still_running + new_halted * remainders

            data = l(data, mask, *args, **kwargs)

            # The Universal Transformer variant have a bug in weighting: they multiply the old weights by
            # 1-update_weights. Since update_weights form a probability distribution anyways, this is not necessary.
            # Moreover the UT way is different from what is described in the ACT paper.
            out_data = ((out_data * (1 - update_weights)) if self.ut_variant else out_data) + update_weights * data

            if still_running.sum() == 0:
                break

        # Calculate length satistics
        if not self.training and has_mask:
            self.update_len_stat(n_updates.squeeze(-1).max(1)[0], mask.src_length_mask.shape[1] - \
                                                                  mask.src_length_mask.int().sum(1))

        self.add_reg(lambda: remainders.mean() * self.act_loss_weight)
        return out_data

    def get_len_stats(self) -> torch.Tensor:
        res = {l: v["val"] / v["sum"] for l, v in self.len_stats.items()}
        self.len_stats = {}
        return res


def ACTTransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: ACTTransformerEncoder(layer, *args, **kwargs)
