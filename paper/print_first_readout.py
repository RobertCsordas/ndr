from print_perf_table_ctl import print_table
from collections import OrderedDict
import lib
from lib.common import group

g = {}
g.update(group(lib.get_runs(["ctl_baselines_transformer_11_first_col"], {"config.state_size": 128}), ["ctl.reversed", "transformer.variant"]))
g.update({k+"_noabs": v for k, v in group(lib.get_runs(["ctl_ndr_residual_first_col"]), ["ctl.reversed", "transformer.variant"]).items()})
print(g.keys())


model_list = OrderedDict()
model_list["transformer.variant_universal"] = "Transformer"
model_list["transformer.variant_relative_universal"] = "\\quad + rel"
model_list["transformer.variant_ndr_residual_noabs"] = "\\quad + rel + gate"

print_table(g, model_list, bf=False)
