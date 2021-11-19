# from print_perf_table import print_table
from collections import OrderedDict
import lib
from lib.common import group
import matplotlib.pyplot as plt

g = {}
runs = lib.get_runs(["ctl_act"], {"config.state_size": 128})
g.update(group([r for r in runs if r.config["act.ut_variant"]==0], ["ctl.reversed", "transformer.variant"]))
g.update({k+"_ut": v for k, v in group([r for r in runs if r.config["act.ut_variant"]==1], ["ctl.reversed", "transformer.variant"]).items()})
print(g.keys())


runs = g['ctl.reversed_1/transformer.variant_act_relative_universal']

s = lib.common.calc_stat({"a": runs}, lambda n: n.startswith("validation/ponder_times"))["a"]
s = {int(k.split("_")[-1]): v for k, v in s.items()}

ticks = list(sorted(s.keys()))
y = [s[i].get() for i in ticks]
x = list(range(0, len(y)))

# print(y)

fig = plt.figure(figsize=[4.5,1.2])
plt.bar(x, height=[a.mean for a in y], yerr=[a.std for a in y], align='center')

plt.xticks(x, ticks)
plt.ylim(0,15)
plt.xlim(x[0]-0.5, x[-1]+0.5)
plt.hlines(runs[0].config["transformer.encoder_n_layers"], x[0]-0.5, x[-1]+0.5, colors="red", linestyles="dotted")
plt.ylabel("Iteration")
plt.xlabel("Sequence length")
fig.savefig("ponder_act_relative.pdf", bbox_inches='tight', pad_inches = 0.01)