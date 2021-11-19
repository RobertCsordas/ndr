#!/usr/bin/env python3
import lib
from lib.common import group, calc_stat
from lib.stat_tracker import StatTracker, Stat
from collections import OrderedDict
from typing import List, Dict, Any


IID_KEY = "validation/iid/accuracy/total"
GEN_KEY = "validation/deeper/accuracy/total"
VALID_KEY = "validation/valid/accuracy/total"

highlight_threshold = 0.02

g = {}

g.update(group(lib.get_runs(["simple_arithmetics_rnn"]), ["seq_classifier.rnn"]))
g.update(group(lib.get_runs(["simple_arithmetics_geometric_no_norm"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["simple_arithmetics_universal"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["simple_arithmetics_gated"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["simple_arithmetics_reltrafo_kaiming"]), ["transformer.variant"]))
print(g.keys())

model_list = OrderedDict()
model_list["seq_classifier.rnn_lstm"] = "LSTM"
model_list["seq_classifier.rnn_bilstm"] = "Bidirectional LSTM"
model_list["asd"] = None
model_list["transformer.variant_universal"] = "Transformer"
model_list["transformer.variant_relative_universal"] = "\\quad + rel"
model_list["transformer.variant_ndr_residual"] = "\\quad + abs/rel + gate"
model_list["transformer.variant_ndr_geometric"] = "\\quad + geom. att. + gate (NDR)"

sgen = lib.CrossValidatedStats(GEN_KEY, VALID_KEY)(g)
siid = calc_stat(g, lambda k: k in {IID_KEY})

sgen50k = lib.CrossValidatedStats(GEN_KEY, VALID_KEY, ["iteration"])(g, test = lambda x: x["iteration"] <= 50000)

def format_res(r):
    # r = r.get()
    return f"{r.mean:.2f} $\\pm$ {r.std:.2f}"

print("Model & IID -> & IID <- & Longer -> & Longer <- \\\\")

rows = []
for k, name in model_list.items():
    if name is None:
        rows.append(None)
        continue
    line = []
    line.append(siid[k][IID_KEY].get())

    line.append(sgen[k])
    line.append(sgen50k[k])
    rows.append(line)


maxcol = [max([r[c].mean for r in rows if r]) for c in range(len(rows[0]))]

print("\\midrule")
for name, line in zip(model_list.values(), rows):
    if line is None:
        print("\\midrule")
        continue

    fline = [format_res(c) for ci, c in enumerate(line)]
    fline = [f"\\bf{{{fc}}}" if (maxcol[ci] - line[ci].mean) < highlight_threshold else fc for ci, fc in enumerate(fline)]
    print(name+" & " + " & ".join(fline) + " \\\\")

print("\\bottomrule")