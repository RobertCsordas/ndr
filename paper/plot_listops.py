from typing import OrderedDict
import lib
from run_model import LoadedModel
import math
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvas
import numpy as np
import math
import matplotlib.patches as patches
import time
import torch

cmap = 'viridis'

def patch(args):
    args["listops.variant"] = "official"

model = LoadedModel("user/project/runid", 100000, patch)

sample = "[SM [MED [MIN 1 7 4 [MAX 2 4 0 8 9 ] ] 7 ] 5 [MED 8 5 8 ] 0 7 ]"

model.run(sample.split(" "), "0")


att_max = model.plots['validation_plots/hack/activations/trafo.layer.att/attention_max'].map
att_head_13 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_13'].map
att_head_6 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_6'].map
att_head_5 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_5'].map
att_head_7 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_7'].map
att_head_3 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_3'].map
att_head_14 = model.plots['validation_plots/hack/activations/trafo.layer.att/head_14'].map
att_op_relationships = model.plots['validation_plots/hack/activations/trafo.layer.att/head_13'].map
gate = model.plots['validation_plots/hack/activations/trafo.layer/gate'].map
labels = model.plots['validation_plots/hack/activations/trafo.layer.att/attention_max'].x_marks

def plot_attention(ax, att, steplabels, title, extra_args=dict(vmin=0, vmax=1)):
    im = ax.imshow(att, interpolation='nearest', aspect='auto', cmap=extra_args.get("cmap",cmap), **extra_args)

    ax.set_xticks(np.arange(att.shape[-1]))
    ax.set_xticklabels(steplabels, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")

    ax.set_yticks(np.arange(att.shape[-1]))
    ax.set_yticklabels(steplabels, fontsize=6, rotation=0, ha="right", rotation_mode="anchor")

    ax.set_title(title, fontsize=8, pad=0.001)
    return im

def plot_all_attentions(att_max, steplabels, fname, nr=4, toffset=0, extra_args=dict(vmin=0, vmax=1)):
    nc = int(math.ceil(att_max.shape[0]/nr))

    figure, ax = plt.subplots(nr, nc, figsize=[nc*3, nr * 3])

    for i in range(att_max.shape[0]):
        im = plot_attention(ax[ i//nc, i % nc], att_max[i], steplabels, f"$t={i+toffset}$", extra_args=extra_args)

    for i in range(att_max.shape[0], nr*nc):
        ax[i//nc, i % nc].set_axis_off()

    # divider = make_axes_locatable(ax[att_max.shape[0] // nc, att_max.shape[0] % nc])
    # cax = divider.append_axes('left', size="7%", pad=0.2,)
    # figure.colorbar(im, cax=cax, pad=0.01)

    # figure.colorbar(im, ax=ax[-1,-1], pad=0.01, location="left")
    # cbar.ax.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)

def plot_gate(a, g, steplabels, title, extra_args=dict(vmin=0, vmax=1)):
    im = a.imshow(g, interpolation='nearest', aspect='auto', cmap=extra_args.get("cmap",cmap), **extra_args)
    a.axes.yaxis.set_visible(False)
    a.set_xticks(np.arange(g.shape[-1]))
    a.set_xticklabels(steplabels, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    a.set_title(title, fontsize=8, pad=0.001)
    return im

def plot_all_gates(gates, steplabels, fname, nr=4,  toffset=0, extra_args=dict(vmin=0, vmax=1)):
    nc = int(math.ceil(gates.shape[0]/nr))

    figure, ax = plt.subplots(nr, nc, figsize=[nc*3, nr * 3])

    for i in range(gates.shape[0]):
        im = plot_gate(ax[ i//nc, i % nc], gates[i], steplabels, f"$t={i+toffset}$", extra_args=extra_args)

    for i in range(gates.shape[0], nr*nc):
        ax[i//nc, i % nc].set_axis_off()

    # divider = make_axes_locatable(ax[gates.shape[0] // nc, gates.shape[0] % nc])
    # cax = divider.append_axes('left', size="7%", pad=0.2,)
    # figure.colorbar(im, cax=cax, pad=0.01)

    # figure.colorbar(im, ax=ax[-1,-1], pad=0.01, location="left")
    # cbar.ax.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)


plot_all_attentions(att_max, labels, "listops_all_att.pdf", nr=6)
plot_all_gates(gate, labels, "listops_all_gate.pdf", nr=6)



def put_box(ax, x,y, w,h, color):
    rect = patches.Rectangle((x-0.5,y-0.5), w, h, fill=False, color=color, linewidth=2)
    ax.add_patch(rect)

    # ax.vlines(3, 7, 26, colors='gray', linestyles='dotted')
    if h > 1:
        ax.hlines(y-0.5, 0, x-1, colors='gray', linestyles='dotted')
        ax.hlines(y + h -0.5, 0, x-1, colors='gray', linestyles='dotted')
    else:
        ax.hlines(y, 0, x-1, colors='gray', linestyles='dotted')

    if w > 1:
        ax.vlines(x - 0.5, y+h, 26, colors='gray', linestyles='dotted')
        ax.vlines(x + w - 0.5, y+h, 26, colors='gray', linestyles='dotted')
    else:
        ax.vlines(x, y+h, 26, colors='gray', linestyles='dotted')


figure, ax = plt.subplots(2, 4, figsize=[4*3, 2*3])


# 0, 2, 3, 11
plot_attention(ax[0,0], att_head_13[0], labels, "head 13, $t=0$")
put_box(ax[0,0], 3, 4, 1, 3, 'red')
put_box(ax[0,0], 7, 8, 1, 6, 'red')
put_box(ax[0,0], 18, 19, 1, 4, 'red')
plot_attention(ax[0,1], att_head_13[2], labels, "head 13, $t=2$")
put_box(ax[0,1], 3, 7, 1, 1, 'red')
plot_attention(ax[0,2], att_head_13[3], labels, "head 13, $t=3$")
put_box(ax[0,2], 3, 7, 1, 1, 'red')
plot_attention(ax[0,3], att_head_13[6], labels, "head 13, $t=6$")
put_box(ax[0,3], 2, 4, 1, 1, 'red')
put_box(ax[0,3], 2, 15, 1, 2, 'red')

plot_attention(ax[1,0], att_head_5[1], labels, "head 5, $t=1$")
put_box(ax[1,0], 12, 7, 1, 1, 'red')
plot_attention(ax[1,1], att_head_7[2], labels, "head 7, $t=2$")
put_box(ax[1,1], 19, 18, 1, 1, 'red')
plot_attention(ax[1,2], att_head_3[6], labels, "head 3, $t=6$")
put_box(ax[1,2], 4, 3, 1, 1, 'red')
plot_attention(ax[1,3], np.maximum(att_head_3, att_head_5)[7], labels, "max(head 3, head 5), $t=7$")
# plot_attention(ax[1,3], (att_head_3 + att_head_5)[7]/2, labels, "max(head 3, head 5), $t=7$")
put_box(ax[1,3], 4, 2, 1, 1, 'red')
put_box(ax[1,3], 15, 2, 1, 1, 'red')

figure.tight_layout()


figure.savefig("listops_annot2.pdf", bbox_inches='tight', pad_inches = 0.01)
