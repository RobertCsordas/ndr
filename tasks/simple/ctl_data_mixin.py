import dataset
import framework
from .. import args


@args
def a(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-ctl.reversed", default=True)
    parser.add_argument("-ctl.max_depth", default=5)
    parser.add_argument("-ctl.n_more_depth", default=5)
    parser.add_argument("-ctl.n_more_depth_valid", default=3)
    parser.add_argument("-ctl.detailed_output", default=True)


class CTLDataMixin:
    VALID_NUM_WORKERS = 0
    
    def get_ctl_loader(self, split: str) -> dataset.CompositionalTableLookup:
        return dataset.CompositionalTableLookup(split, self.helper.args.ctl.max_depth, 11,
                                                reversed=self.helper.args.ctl.reversed,
                                                n_more_depth=self.helper.args.ctl.n_more_depth,
                                                atomic_input=True, copy_input=True, max_n_sampes=1000000,
                                                n_more_depth_valid=self.helper.args.ctl.n_more_depth_valid,
                                                detailed_output=self.helper.args.ctl.detailed_output)

    def create_datasets(self):
        self.batch_dim = 1
        self.train_set = self.get_ctl_loader("train")
        self.valid_sets.valid = self.get_ctl_loader("valid")
        self.valid_sets.test = self.get_ctl_loader("test") 


class CTLClassifierDataMixin(CTLDataMixin):
    def get_ctl_loader(self, split: str) -> dataset.CompositionalTableLookup:
        return dataset.CompositionalTableLookupClassification(split, self.helper.args.ctl.max_depth, 11,
                                                              reversed=self.helper.args.ctl.reversed,
                                                              n_more_depth=self.helper.args.ctl.n_more_depth,
                                                              n_more_depth_valid=self.helper.args.ctl.n_more_depth_valid,
                                                              atomic_input=True, max_n_sampes=1000000)
