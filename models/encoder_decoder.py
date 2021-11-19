import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass


def add_eos(input: torch.Tensor, lengths: torch.Tensor, eos_id: int, batch_dim: int = 1):
    time_dim = 1 - batch_dim
    input = torch.cat((input, torch.zeros_like(input.select(time_dim, 0).unsqueeze(time_dim))), dim=time_dim)
    input.scatter_(time_dim, lengths.unsqueeze(time_dim).long(), value=eos_id)
    return input


def add_front_tokens(input: torch.Tensor, token_id: int, n: int) -> torch.Tensor:
    # input shape: [T, B]
    s = list(input.shape)
    s[0] = n
    return torch.cat((torch.full(s, dtype=input.dtype, device=input.device, fill_value=token_id), input), dim=0)


def add_sos(input: torch.Tensor, sos_id: int) -> torch.Tensor:
    # input shape: [T, B]
    return add_front_tokens(input, sos_id, 1)


@dataclass
class EncoderState:
    outputs: torch.Tensor
    state: Any


class Encoder(torch.nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int, n_layers: int, embedding_size: int, dropout: float,
                 lstm: torch.nn.Module):
        super().__init__()

        self.vocabulary_size = vocabulary_size
        self.dropout = dropout
        self.construct(hidden_size, n_layers, embedding_size, lstm)

    def construct(self, hidden_size: int, n_layers: int, embedding_size: int, lstm: torch.nn.Module):
        self.embedding = torch.nn.Embedding(self.vocabulary_size + 1, embedding_size, 1)
        self.lstm = lstm(embedding_size, hidden_size, n_layers, dropout=self.dropout)

    def set_dropout(self, dropout: float):
        self.dropout = dropout
        self.lstm.dropout = dropout

    def run(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        net = self.embedding(input.long())
        net = F.dropout(net, self.dropout, training=self.training)
        net, state = self.lstm(net, lengths=lengths)
        return net, state

    def __call__(self, inputs: torch.Tensor, lengths: torch.Tensor) -> EncoderState:
        inputs = add_eos(inputs, lengths, self.vocabulary_size)
        return EncoderState(*self.run(inputs, lengths + 1))

    def get_run_summary_state(self, state: EncoderState, lengths: torch.Tensor) -> torch.Tensor:
        d = state.outputs
        d = d.gather(0, lengths.view([1, lengths.shape[0], 1]).expand(-1, -1, d.shape[-1]) - 1).squeeze(0)
        return d


class BidirectionalLSTMEncoder(Encoder):
    def __init__(self, vocabulary_size: int, hidden_size: int, n_layers: int, embedding_size: int, dropout: float,
                 lstm: torch.nn.Module):
        assert hidden_size % 2 == 0
        lstmb = lambda input_size, hidden_size, *args, **kwargs: lstm(input_size, hidden_size // 2, *args, **kwargs, \
                                                                      bidirectional=True)
        super().__init__(vocabulary_size, hidden_size, n_layers, embedding_size, dropout, lstmb)

    def get_run_summary_state(self, state: EncoderState, lengths: torch.Tensor) -> torch.Tensor:
        return state.state[0].view(2, -1, *state.state[0].shape[1:])[:, -1].permute(1, 0, 2).flatten(1)
