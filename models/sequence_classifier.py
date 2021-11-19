import torch
import torch.nn
import torch.nn.functional as F
from typing import Optional
from layers import CudaLSTM
from .encoder_decoder import Encoder
from .helpers import add_thinkig_steps


class SequenceClassifier(torch.nn.Module):
    def __init__(self, in_vocabulary_size: int, n_classes: int, hidden_size: int, n_layers: int,
                 embedding_size: Optional[int], dropout: float, lstm: torch.nn.Module = CudaLSTM, encoder=Encoder,
                 n_thinking_steps: int = 0):

        super().__init__()
        self.dropout = dropout
        self.n_thinking_steps = n_thinking_steps
        self.think_token = in_vocabulary_size
        self.encoder = encoder(in_vocabulary_size + int(n_thinking_steps > 0), hidden_size, n_layers, embedding_size or hidden_size,
                               dropout, lstm)
        self.classifier = torch.nn.Linear(hidden_size, n_classes)

    def __call__(self, input: torch.Tensor, in_lengths: torch.Tensor) -> torch.Tensor:
        # Input: [T, N]
        input, in_lengths = add_thinkig_steps(input, in_lengths, self.n_thinking_steps, self.think_token)

        d = self.encoder(input, in_lengths)
        d = self.encoder.get_run_summary_state(d, in_lengths)
        return self.classifier(d)
