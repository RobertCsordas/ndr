import dataset
from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .listops_data_mixin import ListopsDataMixin
from .. import task


@task()
class ListopsTrafo(ListopsDataMixin, TransformerClassifierMixin, SimpleTask):
    pass
