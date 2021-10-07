import torch
import torch.nn
import torch.nn.functional as F
from .transformer import TransformerDecoderBase, ActivationFunction, TransformerEncoderLayer, TransformerDecoderLayer, \
                         Transformer


class UniversalTransformerEncoder(torch.nn.Module):
    def __init__(self, layer, n_layers: int, *args, **kwargs):
        super().__init__()
        self.layer = layer(*args, **kwargs)
        self.set_n_layers(n_layers)

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


class UniversalTransformerDecoder(TransformerDecoderBase):
    def __init__(self, layer, n_layers: int, d_model: int, *args, **kwargs):
        super().__init__(d_model)
        self.layer = layer(d_model, *args, **kwargs)
        self.set_n_layers(n_layers)

    def set_n_layers(self, n_layers: int):
        self.layers = [self.layer] * n_layers

    def forward(self, data: torch.Tensor, *args, **kwargs):
        for l in self.layers:
            data = l(data, *args, **kwargs)
        return data


def UniversalTransformerEncoderWithLayer(layer=TransformerEncoderLayer):
    return lambda *args, **kwargs: UniversalTransformerEncoder(layer, *args, **kwargs)


def UniversalTransformerDecoderWithLayer(layer=TransformerDecoderLayer):
    return lambda *args, **kwargs: UniversalTransformerDecoder(layer, *args, **kwargs)


class UniversalTransformer(Transformer):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: ActivationFunction = F.relu, attention_dropout: float = 0.0):

        super().__init__(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, activation,
                         UniversalTransformerEncoderWithLayer(),
                         UniversalTransformerDecoderWithLayer(), attention_dropout)
