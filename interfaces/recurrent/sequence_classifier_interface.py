import torch
import torch.nn.functional as F
from typing import Dict
from ..result import RecurrentResult
from ..model_interface import ModelInterface


class SequenceClassifierInterface(ModelInterface):
    def __init__(self, model, transpose_batch_time: bool = False):
        self.model = model
        self.transpose_batch_time = transpose_batch_time

    def decode_outputs(self, outputs: RecurrentResult) -> torch.Tensor:
        return outputs.outputs

    def __call__(self, data: Dict[str, torch.Tensor]) -> RecurrentResult:
        d = data["in"]
        if self.transpose_batch_time:
            d = d.transpose(0, 1)
        outs = self.model(d, data["in_len"].long())
        loss = F.cross_entropy(outs, data["out"].long(), reduction='mean')
        return RecurrentResult(outs, loss)
