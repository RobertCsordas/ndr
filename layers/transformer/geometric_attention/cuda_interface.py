import os
import torch
import multiprocessing
from framework.utils import LockFile
from typing import Tuple, Optional
import torch.nn.functional as F

# Just in time import
# https://pytorch.org/tutorials/advanced/cpp_extens

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'cuda_interface.cu')
outdir = "./cache/geometric_attention"
os.makedirs(outdir, exist_ok=True)

cuda_log_sigmoid_backward = None
cuda_log_sigmoid_forward = None
cuda_window_sum_forward = None
cuda_window_sum_backward = None

def load_extension():
    global cuda_log_sigmoid_forward, cuda_log_sigmoid_backward
    global cuda_window_sum_forward, cuda_window_sum_backward
    if cuda_log_sigmoid_forward is not None:
        return

    with LockFile(outdir + "/lock"):
        from torch.utils.cpp_extension import load

        os.environ["MAX_JOBS"] = str(multiprocessing.cpu_count())
        ext = load(
            extra_cuda_cflags=['--ftemplate-depth=1024'],
            name="geometric_attention_cuda_interface",
            sources=[filename], verbose=True)
            #, build_directory=outdir)

        cuda_log_sigmoid_forward = ext.cuda_log_sigmoid_forward
        cuda_log_sigmoid_backward = ext.cuda_log_sigmoid_backward
        cuda_window_sum_forward = ext.cuda_window_sum_forward
        cuda_window_sum_backward = ext.cuda_window_sum_backward


class LogSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.detach().contiguous()
        ctx.save_for_backward(x)
        a, b = cuda_log_sigmoid_forward(x)
        return a, b
        # return res_a.view_as(x), res_b.view_as(x)

    @staticmethod
    def backward(ctx, grad_in_sigm: torch.Tensor, grad_in_one_minus: torch.tensor) -> torch.Tensor:
        xf, = ctx.saved_tensors
        ga = grad_in_sigm.contiguous()
        gb = grad_in_one_minus.contiguous()
        return cuda_log_sigmoid_backward(xf, ga, gb)[0]


class WindowSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, csum: torch.Tensor, offset: int) -> torch.Tensor:
        ctx.saved_offset = offset
        c2 = csum.detach().contiguous().flatten(end_dim=-3)
        res = cuda_window_sum_forward(c2, offset)
        return res.view_as(csum)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        offset = ctx.saved_offset
        go = grad_output.contiguous().flatten(end_dim=-3)
        res = cuda_window_sum_backward(go, offset)
        return res.view_as(grad_output), None


def window_sum(x: torch.Tensor, offset: int) -> torch.Tensor:
    load_extension()
    return WindowSumFunction.apply(x, offset)


def log_sigmoid(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    load_extension()
    return LogSigmoidFunction.apply(x)


def geometric_attention_activation(logits: torch.Tensor, mask: Optional[torch.Tensor] = None, pos_offset: int = 0,
                                   normalize: bool = True) -> torch.Tensor:
        p, one_minus_p = log_sigmoid(logits)
        not_previos = window_sum(one_minus_p.cumsum(-1), pos_offset)

        probs = (not_previos + p).exp()

        # return probs
        return F.normalize(probs, 1, -1) if normalize else probs