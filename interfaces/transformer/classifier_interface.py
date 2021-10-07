import torch
import torch.nn
from typing import Dict, Tuple
from models.encoder_decoder import add_eos
from ..model_interface import ModelInterface
import framework

from ..result import FeedforwardResult


class TransformerClassifierInterface(ModelInterface):
    def __init__(self, model: torch.nn.Module, label_smoothing: float = 0.0):
        self.model = model
        self.label_smoothing = label_smoothing

    def decode_outputs(self, outputs: FeedforwardResult) -> torch.Tensor:
        return outputs.outputs

    def __call__(self, data: Dict[str, torch.Tensor]) -> FeedforwardResult:
        res = self.model(data["in"].transpose(0, 1), data["in_len"].long())
        loss = framework.layers.cross_entropy(res, data["out"], reduction='mean', smoothing=self.label_smoothing)
        return FeedforwardResult(res, loss)
