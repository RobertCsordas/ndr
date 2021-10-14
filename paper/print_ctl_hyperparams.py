#!/usr/bin/env python3
import lib
from lib.common import group, calc_stat
from collections import OrderedDict

g = group(lib.get_runs(["baselines_rnn"]), ["ctl.reversed", "seq_classifier.rnn"])
g.update(group(lib.get_runs(["baselines_transformer_11"], {"config.state_size": 128}), ["ctl.reversed", "transformer.variant"]))
g.update({k+"_noabs": v for k, v in group(lib.get_runs(["ndr_no_absgate"]), ["ctl.reversed", "transformer.variant"]).items()})
g.update({k+"_abs": v for k, v in group(lib.get_runs(["ndr"]), ["ctl.reversed", "transformer.variant"]).items()})
g.update(group(lib.get_runs(["ndr_geometric"]), ["ctl.reversed", "transformer.variant"]))
g.update(group(lib.get_runs(["baselines_transformer_geometric"]), ["ctl.reversed", "transformer.variant"]))
print(g.keys())


model_list = OrderedDict()
model_list["seq_classifier.rnn_lstm"] = "LSTM"
model_list["seq_classifier.rnn_dnc"] = "DNC"
model_list["asd"] = None
model_list["transformer.variant_universal"] = "Transformer"
model_list["transformer.variant_relative_universal"] = "\quad + rel"
model_list["transformer.variant_ndr_residual_noabs"] = "\quad + rel + gate"
model_list["transformer.variant_ndr_residual_abs"] = "\quad + abs/rel + gate"
model_list["transformer.variant_geometric_transformer"] = "\quad + geom. att."
model_list["transformer.variant_ndr_geometric"] = "\quad + geom. att. + gate"

def is_trafo(config):
    return config["task"] not in {"ctl_rnn_classifier"}

def fix_decimals(f):
    fs = str(round(f, 1))
    if "." in fs:
        fs = fs.split(".")
        fs[1] = fs[1].rstrip("0")
        if not fs[1]:
            fs = fs[0]
        else:
            fs = ".".join(fs)
    return fs
    


hparams = OrderedDict()
hparams["$d_{\\text{model}}$"] = lambda config: config["state_size"]
hparams["$d_{\\text{FF}}$"] = lambda config: config["state_size"] * config["transformer.ff_multiplier"] if is_trafo(config) else "-"
hparams["$n_\\text{heads}$"] = lambda config: config["transformer.n_heads"] if is_trafo(config) else "-"
hparams["$n_\\text{layers}$"] = lambda config: config["transformer.encoder_n_layers"] if is_trafo(config) else config["n_layers"]
hparams["batch s."] = lambda config: config["batch_size"]
hparams["learning rate"] = lambda config: f"${fix_decimals(config['lr']*10000)}*10^{{-4}}$"
hparams["weight d."] = lambda config: config["wd"] if config["wd"] else "-"
hparams["dropout"] = lambda config: config["dropout"] if config["dropout"] else "-"
hparams["$n_\\text{iters}$"] = lambda config: f"{config['stop_after']//1000}k"
# hparams["grad_clip"] = lambda config: config["grad_clip"] if config["grad_clip"] else "-"

print("\\toprule")
print("& " + " & ".join(hparams.keys())+" \\\\")
print("\\midrule")
for k, name in model_list.items():
    if name is None:
        print("\\midrule")
        continue

    run = g[f"ctl.reversed_0/{k}"][0]
    cols = [str(hf(run.config)) for _, hf in hparams.items()]

    print(name + " & " + " & ".join(cols)+" \\\\")
print("\\bottomrule")