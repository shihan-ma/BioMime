import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from easydict import EasyDict as edict
from matplotlib.collections import LineCollection


def plot_muaps(muaps, path, cfg=None):
    """
    muaps: [arr]
    """

    if cfg == None:
        cfg = edict({'cmap': 'viridis', 'figsize': [15, 9], 'linewidth': 0.8, 'alpha': 0.8})

    if not isinstance(muaps, np.ndarray):
        muaps = muaps.numpy()

    num_muaps, steps, n_row, n_col, n_t = muaps.shape
    x_ = np.linspace(0, 1, n_t)
    cmap = plt.get_cmap(cfg.cmap)
    colors = []
    for i in np.linspace(1, 0, steps):
        colors.append(list(cmap(i)[:3]))

    for i in range(num_muaps):
        cur_muaps = -muaps[i]
        p_max_amp = np.max(cur_muaps[:])
        n_max_amp = np.min(cur_muaps[:])
        fig, axes = plt.subplots(n_row, n_col, figsize=tuple(cfg.figsize))
        for row in range(n_row):
            for col in range(n_col):
                segs = [np.column_stack([x_, cur_muaps[sp, row, col]]) for sp in range(steps)]
                line_segments = LineCollection(segs, array=x_, colors=colors, linewidths=(cfg.linewidth), alpha=cfg.alpha)
                axes[row, col].add_collection(line_segments)
                axes[row, col].set_ylim([n_max_amp, p_max_amp])
                axes[row, col].set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(path, 'muap_{}.jpg'.format(i)))
        plt.savefig(os.path.join(path, 'muap_{}.svg'.format(i)))
        plt.close()


def save_dynamic_muaps_video(muaps, res_path):
    nf, w, h, t = muaps.shape
    for f in range(nf):
        cur_muap = -muaps[f]
        vmin, vmax = np.min(cur_muap[:]), np.max(cur_muap[:])
        fig, axes = plt.subplots(w, h, figsize=(64, 48))
        for row in range(w):
            for col in range(h):
                axes[row, col].plot(cur_muap[row, col], linewidth=2, color='k')
                axes[row, col].set_ylim([vmin, vmax])
                axes[row, col].axis('off')
        plt.savefig(res_path + 'dynamic_muaps_f{}.jpg'.format(f))
        plt.close()
    command_list = ['ffmpeg', '-r', '5', '-i', res_path + 'dynamic_muaps_f%d.jpg', '-vcodec', 'mpeg4', res_path + 'dynamic_muaps.mp4']
    if subprocess.run(command_list).returncode == 0:
        print('video saved...')
    else:
        print('Error when saving video...')


def save_one_muap_video(muap, res_path):
    w, h, t = muap.shape
    vmin, vmax = np.min(muap[:]), np.max(muap[:])
    for f in range(t):
        fig, axes = plt.subplots()
        axes.imshow(muap[:, :, f], (w, h), interpolation='bicubic', vmin=vmin, vmax=vmax)
        axes.axis('off')
        plt.savefig(res_path + 'muap_t{}.jpg'.format(f))
        plt.close()
    command_list = ['ffmpeg', '-r', '5', '-i', res_path + 'muap_t%d.jpg', '-vcodec', 'mpeg4', res_path + 'muap_potential.mp4']
    if subprocess.run(command_list).returncode == 0:
        print('video saved...')
    else:
        print('Error when saving video...')
