import torch
import torch.nn
from typing import Dict, Any, Callable


class RegularizedLayer:
    def __init__(self) -> None:
        super().__init__()
        self.reg_accumulated = 0
        self.regularization_present = False

    @property
    def reg_enabled(self) -> bool:
        return self.training and self.regularization_present

    def add_reg(self, l: Callable[[], torch.Tensor]):
        if self.reg_enabled:
            self.reg_accumulated = self.reg_accumulated + l()

    def get_reg_loss(self) -> torch.Tensor:
        rl = self.reg_accumulated
        self.reg_accumulated = 0
        return rl


class LayerRegularizer:
    def __init__(self, module: torch.nn.Module, options: Dict[str, Any] = {}):
        self.modules = []
        for n, m in module.named_modules():
            if isinstance(m, RegularizedLayer):
                self.modules.append((n, m))
                m.regularization_present = True

    def get(self) -> torch.Tensor:
        res = 0
        for _, m in self.modules:
            res = res + m.get_reg_loss()

        return res
