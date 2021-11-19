#!/usr/bin/env python3
import lib
from lib.common import group, calc_stat
from lib.stat_tracker import StatTracker, Stat
from collections import OrderedDict
from typing import List, Dict, Any


IID_KEY = "validation/iid/accuracy/total"
GEN_KEY = "validation/depth/accuracy/total"
VALID_KEY = "validation/valid/accuracy/total"

highlight_threshold = 0.02

g = {}

g.update(group(lib.get_runs(["listops_big_lowlr"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["listops_big_reluni"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["listops_big_uni"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["listops_big_lowlr_ndr_daint"]), ["transformer.variant"]))
g.update(group(lib.get_runs(["listops_big_lstm"]), ["seq_classifier.rnn"]))
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
    # line.append(format_res(sgen50k[k]))
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