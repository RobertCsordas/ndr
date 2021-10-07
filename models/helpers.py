import torch
import torch.nn.functional as F


def add_thinkig_steps(input: torch.Tensor, in_lengths: torch.Tensor, n_thinking_steps: int, think_token: int) ->\
        torch.Tensor:

    if n_thinking_steps <= 0:
        return input, in_lengths

    m = torch.arange(0, input.shape[0], dtype=torch.long, device=input.device).unsqueeze(1) >= \
        in_lengths.unsqueeze(0)

    input = input.masked_fill(m, think_token)
    return F.pad(input, (0, 0, 0, n_thinking_steps), value=think_token), in_lengths + n_thinking_steps
