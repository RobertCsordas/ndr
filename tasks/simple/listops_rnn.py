import torch.nn
import torch.utils.data
from .simple_task import SimpleTask
from .sequence_classifier_mixin import SequenceClassifierMixin
from ..import task
from .listops_data_mixin import ListopsDataMixin


@task()
class ListopsRNN(SequenceClassifierMixin, ListopsDataMixin, SimpleTask):
    pass
