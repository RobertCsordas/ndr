import torch.nn
import torch.utils.data
from .simple_task import SimpleTask
from .ctl_data_mixin import CTLClassifierDataMixin
from .sequence_classifier_mixin import SequenceClassifierMixin
from ..import task
from .simple_arithmetics_data_mixin import SimpleArithmeticDataMixin


@task()
class SimpleArithmeticsRNN(SequenceClassifierMixin, SimpleArithmeticDataMixin, SimpleTask):
    pass
