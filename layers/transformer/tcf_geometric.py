
import torch
import torch.nn
from layers.layer_with_visualization import LayerWithVisualization
from layers.regularized_layer import RegularizedLayer
from typing import Optional, Dict, Any
import framework
from .direction_sensitive_geometric import DirectionSensitiveGeometricAttentionMyInit, AttentionMask


class TCFGeometric(RegularizedLayer, LayerWithVisualization):
    def __init__(self, d_model: int, nhead: int, dropout: float, scalar_gate: bool = False,
                 attention_dropout=0, p_gate_drop=0.05, dim_feedforward: Optional[int] = None, 
                 global_content_bias: bool = True, normalize_score=True, gate_size_multiplier=1, **kwargs):
        super().__init__()
        self.plot_cache = []

        self.reg_loss = 0

        dim_feedforward = dim_feedforward or (4*d_model)
        self.att = DirectionSensitiveGeometricAttentionMyInit(d_model, nhead, dropout=attention_dropout,
                                                              normalize_score=normalize_score,
                                                              global_content_bias = global_content_bias)

        self.p1 = torch.nn.Linear(d_model, dim_feedforward)
        self.p2 = torch.nn.Linear(dim_feedforward, d_model)

        self.g1 = torch.nn.Linear(d_model, d_model * gate_size_multiplier)
        self.g2 = torch.nn.Linear(d_model * gate_size_multiplier, 1 if scalar_gate else d_model)

        self.nmerge = torch.nn.LayerNorm(d_model)
        self.no = torch.nn.LayerNorm(d_model)

        self.drop = torch.nn.Dropout(dropout)

        self.g2.bias.data.fill_(-3)
        self.p_gate_drop = p_gate_drop

        self.reset_parameters()

    def forward(self, src: torch.Tensor, mask: Optional[AttentionMask] = None) -> torch.Tensor:
        input = self.att(src, src, mask)

        net = self.nmerge(src + self.drop(input))

        mid = self.drop(torch.relu(self.p1(net)))
        proj = self.p2(mid)
        proj = self.no(proj)

        gate = self.g2(self.drop(torch.relu(self.g1(net))))
        bgate = torch.sigmoid(gate)

        if self.training and self.p_gate_drop>0:
            bgate = bgate.masked_fill(torch.rand(*bgate.shape[:-1], 1, device=bgate.device, dtype=bgate.dtype) < self.p_gate_drop, 0)

        if self.visualization_enabled:
            self.plot_cache.append(bgate[0])

        src = src * (1-bgate) + proj * bgate

        return src

    def plot(self, options: Dict[str, Any]) -> Dict[str, Any]:
        r = {}
        if self.visualization_enabled:
            r["gate"] = framework.visualize.plot.AnimatedHeatmap(
                        torch.stack(self.plot_cache, 0).transpose(1,2),
                        ylabel="dest", xlabel="src", textval=False, x_marks=options.get("steplabel"))
            self.plot_cache.clear()

        return r


    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.p1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.p2.weight, gain=torch.nn.init.calculate_gain('tanh'))

        torch.nn.init.xavier_uniform_(self.g1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.g2.weight, gain=torch.nn.init.calculate_gain('sigmoid'))
