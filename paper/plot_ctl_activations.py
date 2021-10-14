import lib
import wandb
import os
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_agg import FigureCanvas
import numpy as np
import math


geometric = "user/project/runid"
baseline = "user/project/runid"
ndr = "user/project/runid"
end_inds = [5, 10]


def load(run, rel_path):
    rel_fname = f"activations/{run.id}/{rel_path}"
    if not os.path.isfile(rel_fname):
        run.file(rel_path).download(root=f"activations/{run.id}", replace=True)
    return torch.load(rel_fname)


def plot_attention(ax, att, steplabels, title):
    im = ax.imshow(att, interpolation='nearest', aspect='auto', cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(np.arange(att.shape[-1]))
    ax.set_xticklabels(steplabels, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")

    ax.set_yticks(np.arange(att.shape[-1]))
    ax.set_yticklabels(steplabels, fontsize=6, rotation=0, ha="right", rotation_mode="anchor")

    ax.set_title(title, fontsize=8, pad=0.001)
    return im



def plot_all_attentions(run, fname, nr=4, file="export/raw_plots/activations/trafo.layer.att/attention_max.pth"):
    att_max = load(run, file)
    steplabels = load(run, "export/raw_plots/steplabels.pth")

    nc = int(math.ceil(att_max.shape[0]/nr))

    figure, ax = plt.subplots(nr, nc, figsize=[nc*2, nr * 2])

    for i in range(att_max.shape[0]):
        im = plot_attention(ax[ i//nc, i % nc], att_max[i], steplabels, f"$t={i}$")

    for i in range(att_max.shape[0], nr*nc):
        ax[i//nc, i % nc].set_axis_off()

    divider = make_axes_locatable(ax[att_max.shape[0] // nc, att_max.shape[0] % nc])
    cax = divider.append_axes('left', size="7%", pad=0.2,)
    figure.colorbar(im, cax=cax, pad=0.01)

    # figure.colorbar(im, ax=ax[-1,-1], pad=0.01, location="left")
    # cbar.ax.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)


cmap = 'viridis'

api = wandb.Api()
# run = api.run(ndr)
run = api.run(geometric)


def plot_gate(a, g, steplabels, title):
    im = a.imshow(g, interpolation='nearest', aspect='auto', cmap=cmap, vmin=0, vmax=1)
    a.axes.yaxis.set_visible(False)
    a.set_xticks(np.arange(g.shape[-1]))
    a.set_xticklabels(steplabels, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
    a.set_title(title, fontsize=8, pad=0.001)
    return im

# end_inds = [gates.shape[-1]-3, gates.shape[-1]-2]
def plot_few_gates(run, fname):
    gates = load(run, "export/raw_plots/activations/trafo.layer/gate.pth")
    steplabels = load(run, "export/raw_plots/steplabels.pth")

    titles = ['$t=0$', '$t=1$', f'$t={end_inds[0]}$', f'$t={end_inds[1]}$']

    figure, ax = plt.subplots(1, 4, figsize=[8,2])

    plot_gate(ax[0], gates[0], steplabels, titles[0])
    plot_gate(ax[1], gates[1], steplabels, titles[1])
    plot_gate(ax[2], gates[end_inds[0]], steplabels, titles[2])
    im=plot_gate(ax[3], gates[end_inds[1]], steplabels, titles[3])

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.25)
    # plt.show()

    cbar = figure.colorbar(im, ax=ax.ravel().tolist(), pad=0.01)
    cbar.ax.tick_params(labelsize=7)

    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)


plot_few_gates(run, "gates.pdf")

def plot_few_activations(run, fname):

    att_max = load(run, "export/raw_plots/activations/trafo.layer.att/attention_max.pth")
    steplabels = load(run, "export/raw_plots/steplabels.pth")

  
    titles = ['$t=0$', '$t=1$', f'$t={end_inds[0]}$', f'$t={end_inds[1]}$']

    figure, ax = plt.subplots(1, 4, figsize=[8,2])

    plot_attention(ax[0], att_max[0], steplabels, titles[0])
    plot_attention(ax[1], att_max[1], steplabels, titles[1])
    plot_attention(ax[2], att_max[end_inds[0]], steplabels, titles[2])
    im=plot_attention(ax[3], att_max[end_inds[1]], steplabels, titles[3])

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.25)
    # plt.show()

    cbar = figure.colorbar(im, ax=ax.ravel().tolist(), pad=0.01)
    cbar.ax.tick_params(labelsize=7)

    figure.savefig("attention.pdf", bbox_inches='tight', pad_inches = 0.01)


def plot_few_gates(run, fname):
    gates = load(run, "export/raw_plots/activations/trafo.layer/gate.pth")
    steplabels = load(run, "export/raw_plots/steplabels.pth")

    titles = ['$t=0$', '$t=1$', f'$t={end_inds[0]}$', f'$t={end_inds[1]}$']

    figure, ax = plt.subplots(1, 4, figsize=[8,2])

    plot_gate(ax[0], gates[0], steplabels, titles[0])
    plot_gate(ax[1], gates[1], steplabels, titles[1])
    plot_gate(ax[2], gates[end_inds[0]], steplabels, titles[2])
    im=plot_gate(ax[3], gates[end_inds[1]], steplabels, titles[3])

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.25)
    # plt.show()

    cbar = figure.colorbar(im, ax=ax.ravel().tolist(), pad=0.01)
    cbar.ax.tick_params(labelsize=7)

    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)


# plot_few_gates(run, "gates.pdf")

def plot_few(run, fname):
    att_max = load(run, "export/raw_plots/activations/trafo.layer.att/attention_max.pth")
    gates = load(run, "export/raw_plots/activations/trafo.layer/gate.pth")
    steplabels = load(run, "export/raw_plots/steplabels.pth")

  
    titles = ['$t=0$', '$t=1$', f'$t={end_inds[0]}$', f'$t={end_inds[1]}$']

    figure, ax = plt.subplots(2, 4, figsize=[1+8, 4])

    plot_attention(ax[0,0], att_max[0], steplabels, titles[0])
    plot_attention(ax[0,1], att_max[1], steplabels, titles[1])
    plot_attention(ax[0,2], att_max[end_inds[0]], steplabels, titles[2])
    im1=plot_attention(ax[0,3], att_max[end_inds[1]], steplabels, titles[3])

    plot_gate(ax[1,0], gates[0], steplabels, titles[0])
    plot_gate(ax[1,1], gates[1], steplabels, titles[1])
    plot_gate(ax[1,2], gates[end_inds[0]], steplabels, titles[2])
    im=plot_gate(ax[1,3], gates[end_inds[1]], steplabels, titles[3])

    figure.tight_layout()
    figure.subplots_adjust(wspace=0.25)
    # plt.show()

    cbar = figure.colorbar(im, ax=ax.ravel().tolist(), pad=0.02, aspect=20)
    cbar.ax.tick_params(labelsize=7)

    # cbar = figure.colorbar(im1, ax=ax.ravel().tolist(), pad=0.01)
    # cbar.ax.tick_params(labelsize=7)

    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)


# def plot_all_gates(run):

# plot_few_activations(run, "activations.pdf")

plot_few(run, "att_and_gates.pdf")


run = api.run(geometric)

plot_all_attentions(run, "all_attention_geometric.pdf")

def plot_all_gates(run, fname, nr=4):
    gates = load(run, "export/raw_plots/activations/trafo.layer/gate.pth")
    steplabels = load(run, "export/raw_plots/steplabels.pth")

    nc = int(math.ceil(gates.shape[0]/nr))

    figure, ax = plt.subplots(nr, nc, figsize=[nc*2, nr * 2])

    for i in range(gates.shape[0]):
        im = plot_gate(ax[ i//nc, i % nc], gates[i], steplabels, f"$t={i}$")

    for i in range(gates.shape[0], nr*nc):
        ax[i//nc, i % nc].set_axis_off()

    divider = make_axes_locatable(ax[gates.shape[0] // nc, gates.shape[0] % nc])
    cax = divider.append_axes('left', size="7%", pad=0.2,)
    figure.colorbar(im, cax=cax, pad=0.01)

    # figure.colorbar(im, ax=ax[-1,-1], pad=0.01, location="left")
    # cbar.ax.tick_params(labelsize=7)

    figure.tight_layout()
    figure.savefig(fname, bbox_inches='tight', pad_inches = 0.01)

plot_all_gates(run, "all_gates_geometric.pdf")

run = api.run(ndr)

plot_all_attentions(run, "all_attention_ndr.pdf")

plot_all_gates(run, "all_attention_ndr.pdf")


plot_all_attentions(api.run(baseline), "all_attention_baseline.pdf", nr=3, file="export/raw_plots/activations/trafo.layer.self_attn/attention_max.pth")
