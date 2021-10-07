import torch
from typing import Tuple, Optional
import framework


class CudaLSTM(torch.nn.LSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # framework.utils.lstm_init(self)

    def get_batch_size(self, state: Tuple[torch.Tensor, torch.Tensor]) -> int:
        return state[0].shape[1]

    def forward(self, input: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if lengths is not None:
            input = torch.nn.utils.rnn.pack_padded_sequence(input, lengths.cpu().long(), enforce_sorted=False)

        o, s2 = super().forward(input, state)
        if lengths is not None:
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)

        return o, s2
