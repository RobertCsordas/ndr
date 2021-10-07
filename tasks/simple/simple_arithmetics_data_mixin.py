import dataset
from .. import task, args
import framework


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-simple_arithmetics.rpn", default=False)
    parser.add_argument("-simple_arithmetics.nnums", default=10)
    parser.add_argument("-simple_arithmetics.n_samples", default=1000000)


class SimpleArithmeticDataMixin:
    VALID_NUM_WORKERS = 0

    VIS_DATASET_FILTER = {"valid", "tiny"}

    def create_datasets(self):
        self.batch_dim = 1
        rpn = self.helper.args.simple_arithmetics.rpn
        nnums = self.helper.args.simple_arithmetics.nnums
        TRAIN_MAX_LEN = 6
        self.train_set = dataset.SimpleArithmetics("train", 1, (2, TRAIN_MAX_LEN), 0.2, 50, self.helper.args.simple_arithmetics.n_samples, rpn=rpn, n_nums=nnums)
        self.valid_sets.iid = dataset.SimpleArithmetics("valid", 1, (2, TRAIN_MAX_LEN), 0.2, 50, 1000, rpn=rpn, n_nums=nnums)
        self.valid_sets.tiny = dataset.SimpleArithmetics("valid", 1, (4, 4), 0.3, 50, 100, rpn=rpn, n_nums=nnums)
        self.valid_sets.valid = dataset.SimpleArithmetics("valid", 1, (TRAIN_MAX_LEN+1, TRAIN_MAX_LEN+1), 0.3, 50, 1000, rpn=rpn, n_nums=nnums)
        self.valid_sets.deeper = dataset.SimpleArithmetics("valid", 1, (TRAIN_MAX_LEN+2, TRAIN_MAX_LEN+3), 0.3, 50, 1000, rpn=rpn, n_nums=nnums)

        print("Length of the training set", len(self.train_set))
        print("In vocabulary: "+", ".join(self.train_set.in_vocabulary.inv_words[a] for a in range(len(self.train_set.in_vocabulary))))
