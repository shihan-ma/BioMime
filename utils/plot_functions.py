import matplotlib.pyplot as plt
import numpy as np
import subprocess


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
