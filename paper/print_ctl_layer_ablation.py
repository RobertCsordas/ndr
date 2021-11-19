#!/usr/bin/env python3
import lib
from lib.common import group, calc_stat
from collections import OrderedDict
import wandb


api = wandb.Api()

g = group(lib.get_runs(["ctl_ndr_layer_ablation"]), ["ctl.reversed", "transformer.encoder_n_layers"])

n_layers = list(reversed(sorted({int(a.split("_")[-1]) for a in g.keys()})))

g = calc_stat(g, lambda x: x in {"validation/test/accuracy/deeper", "validation/valid/accuracy/iid"} )

print(g)

rows = {}
for l in n_layers:
    rows[l] = [g[f"ctl.reversed_1/transformer.encoder_n_layers_{l}"]["validation/valid/accuracy/iid"],
               g[f"ctl.reversed_0/transformer.encoder_n_layers_{l}"]["validation/valid/accuracy/iid"],
               g[f"ctl.reversed_1/transformer.encoder_n_layers_{l}"]["validation/test/accuracy/deeper"],
               g[f"ctl.reversed_0/transformer.encoder_n_layers_{l}"]["validation/test/accuracy/deeper"]]

print("& \\multicolumn{2}{c}{IID} & \\multicolumn{2}{c}{Test} \\\\")
print("\\cmidrule(lr){2-3}  \\cmidrule(lr){4-5}")
print("$n_\\text{layers}$ & Forward & Backward & Forward & Backward \\\\")
print("\\midrule")
for l in n_layers:
    print(f"{l}", end="")
    for c in rows[l]:
        c = c.get()
        print(f" & {c.mean:.2f} $\\pm$ {c.std:.2f}", end="")
    print(" \\\\")
