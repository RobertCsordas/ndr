import torch.nn
import torch.utils.data
from .simple_task import SimpleTask
from .ctl_data_mixin import CTLClassifierDataMixin
from .sequence_classifier_mixin import SequenceClassifierMixin
from ..import task


@task()
class CtlRnnClassifier(SequenceClassifierMixin, CTLClassifierDataMixin, SimpleTask):
    pass
