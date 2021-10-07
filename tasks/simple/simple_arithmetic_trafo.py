import dataset
from .simple_task import SimpleTask
from .transformer_classifier_mixin import TransformerClassifierMixin
from .listops_data_mixin import ListopsDataMixin
from .simple_arithmetics_data_mixin import SimpleArithmeticDataMixin
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-simple_arithmetics.rpn", default=False)
    parser.add_argument("-simple_arithmetics.nnums", default=10)


@task()
class SimpleArithmeticTrafo(SimpleArithmeticDataMixin, TransformerClassifierMixin, SimpleTask):
    pass