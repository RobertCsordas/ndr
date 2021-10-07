from .ctl_data_mixin import CTLClassifierDataMixin
from .transformer_classifier_mixin import TransformerClassifierMixin
from .simple_task import SimpleTask
from typing import Dict, Any, List
from .. import task


@task()
class CTLTrafoClassifier(CTLClassifierDataMixin, TransformerClassifierMixin, SimpleTask):
    pass