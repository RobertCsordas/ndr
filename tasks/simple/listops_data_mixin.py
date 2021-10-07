import dataset
import framework
from .. import args
from typing import Dict, Any

@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-listops.variant", choice=["big", "official"], default="big")


class ListopsDataMixin:
    VALID_NUM_WORKERS = 0

    def create_datasets(self):
        self.batch_dim = 1
        if self.helper.args.listops.variant == "official":
            self.train_set = dataset.transformations.LimitLength(dataset.ListopsOfficial(["train"]), 50)
            self.valid_sets.test_small = dataset.transformations.LimitLength(dataset.ListopsOfficial(["test"]), 25)
            self.valid_sets.valid = dataset.transformations.LimitLength(dataset.ListopsOfficialGenerate(10000), 500)
            self.valid_sets.iid = dataset.transformations.LimitLength(dataset.ListopsOfficial(["test"]), 500)
        elif self.helper.args.listops.variant == "big":
            self.train_set = dataset.ListOps("train", length=50, depth=6, p_op=0.3, n_samples=1000000, equivalize_depdendency_depth=True)
            self.valid_sets.iid = dataset.ListOps("test", length=50, depth=6, p_op=0.3, n_samples=1000, equivalize_depdendency_depth=True)
            self.valid_sets.valid = dataset.ListOps("test", length=50, depth=[7,7], p_op=0.3, n_samples=1000, equivalize_depdendency_depth=True)
            self.valid_sets.depth = dataset.ListOps("test", length=50, depth=[8,9], p_op=0.3, n_samples=1000, equivalize_depdendency_depth=True)
        else:
            assert False, f"Invalid ListOps variant: {self.helper.args.listops.variant}"

        print("Length of the training set", len(self.train_set))
        print("In vocabulary: "+", ".join(self.train_set.in_vocabulary.inv_words[a] for a in range(len(self.train_set.in_vocabulary))))


    def validate(self) -> Dict[str, Any]:
        plots = super().validate()
        # Needed for W & B hyperopt
        if "iid" in self.valid_sets and "valid" in self.valid_sets:
            plots["mix/accuracy/total"] = (plots["iid/accuracy/total"] + plots["valid/accuracy/total"]) / 2
        return plots
