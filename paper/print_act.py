# from print_perf_table import print_table
from collections import OrderedDict
import lib
from lib.common import group
import matplotlib.pyplot as plt

g = {}
runs = lib.get_runs(["ctl_act"], {"config.state_size": 128})
g.update(group([r for r in runs if r.config["act.ut_variant"]==0], ["ctl.reversed", "transformer.variant"]))
g.update({k+"_ut": v for k, v in group([r for r in runs if r.config["act.ut_variant"]==1], ["ctl.reversed", "transformer.variant"]).items()})
g.update(group(lib.get_runs(["ctl_baselines_transformer_11"], {"config.state_size": 128}), ["ctl.reversed", "transformer.variant"]))
print(g.keys())


model_list = OrderedDict()
model_list["transformer.variant_universal"] = "Transformer"
model_list["transformer.variant_act_universal"] = "\\quad + ACT"
model_list["transformer.variant_act_universal_ut"] = "\\quad + ACT-UT"
model_list["asd"] = None
model_list["transformer.variant_relative_universal"] = "\\quad + rel"
model_list["transformer.variant_act_relative_universal"] = "\\quad + rel + ACT"
model_list["transformer.variant_act_relative_universal_ut"] = "\\quad + rel + ACT-UT"
model_list["transformer.variant_geometric_transformer"] = "\\quad + geom. att."
model_list["transformer.variant_act_relative_universal_geometric"] = "\\quad + rel + geom + ACT"
model_list["transformer.variant_act_relative_universal_geometric_ut"] = "\\quad + rel + geom + ACT-UT"


IID_KEY = "validation/valid/accuracy/iid"
GEN_KEY = "validation/test/accuracy/deeper"
VAL_KEY = "validation/valid/accuracy/deeper_val"

highlight_threshold = 0.025

def format_res(r):
    # r = r.get()
    return f"{r.mean:.2f} $\\pm$ {r.std:.2f}"


def print_table(g, model_list, bf=True):
    siid = lib.common.calc_stat(g, lambda k: k in {IID_KEY})
    sgen = lib.CrossValidatedStats(GEN_KEY, VAL_KEY)(g)
    sgen30k = lib.CrossValidatedStats(GEN_KEY, VAL_KEY, ["iteration"])(g, test=lambda x: x["iteration"] <= 30000)

    print("\\midrule")
    rows = []
    for k, name in model_list.items():
        if name is None:
            rows.append(None)
            continue
        line = []
        for r in [1, 0]:
            sthis = siid[f"ctl.reversed_{r}/{k}"]

            line.append(sthis[IID_KEY].get())

        for r in [1, 0]:
            sthis = sgen[f"ctl.reversed_{r}/{k}"]

            line.append(sthis)


        for r in [1, 0]:
            sthis = sgen30k[f"ctl.reversed_{r}/{k}"]

            line.append(sthis)

        rows.append(line)

    maxcol = [max([r[c].mean for r in rows if r]) for c in range(len(rows[0]))]

    for name, line in zip(model_list.values(), rows):
        if line is None:
            print("\\midrule")
            continue

        fline = [format_res(c) for ci, c in enumerate(line)]
        if bf:
            fline = [f"\\bf{{{fc}}}" if (maxcol[ci] - line[ci].mean) < highlight_threshold else fc for ci, fc in enumerate(fline)]
        print(name+" & " + " & ".join(fline) + " \\\\")

    print("\\bottomrule")

print_table(g, model_list, bf=False)
