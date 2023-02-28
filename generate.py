import argparse
import sys
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from utils.basics import update_config, load_generator
from utils.plot_functions import plot_muaps
from utils.prepare_params import num_samples, steps, num, depth, angle, iz, cv, length, changes
from BioMime.generator import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MUAPs')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')

    parser.add_argument('--model_pth', required=True, type=str, help='file of best model')
    parser.add_argument('--mode', default='sample', type=str, help='sample or morph')
    parser.add_argument('--data_path', default='default', type=str, help='file of data to morph')
    parser.add_argument('--res_path', required=True, type=str, help='path of result folder')

    args = parser.parse_args()
    cfg = update_config('./config/' + args.cfg)

    # Check modes
    mode = args.mode
    if mode == 'morph':
        assert args.data_path != 'default', 'Datapath for existing MUAPs required.'
        muaps = np.load(args.data_path)
        num_samples = muaps.shape[0]
    else:
        zi = torch.randn(num_samples, cfg.Model.Generator.Latent)

    # Define muscle labels for each MU if you want to use msk model, List(str), len = num_samples
    # all possible muscle labels: ['ECRL', 'ECRB', 'ECU', 'FCR', 'FCU', 'PL', 'FDSL', 'FDSR', 'FDSM', 'FDSI', 'FDPL', 'FDPR', 'FDPM', 'FDPI', 'EDCL', 'EDCR', 'EDCM', 'EDCI', 'EDM', 'EIP', 'EPL', 'EPB', 'FPL', 'APL', 'APB', 'FPB', 'OPP', 'ADPt', 'ADPo', 'ADM', 'FDM'] in current msk model
    # The most commonly used forearm muscles: ECRL, ECRB, ECU, FCR, FCU, PL, FDS, FDP, EDC, EDM, EPL, FPL
    if len(changes) > 0:
        # ----------- user defined -----------
        ms_labels = ['ECRL', 'ECRB', 'ECU', 'FCU', 'FDSI']
        # ----------- user defined -----------
        assert len(ms_labels) == num_samples
        ch_depth = changes['depth'].loc[:, ms_labels]
        ch_cv = changes['cv'].loc[:, ms_labels]
        ch_ms_lens = changes['len'].loc[:, ms_labels]
        assert ch_depth.shape[0] == steps and ch_cv.shape[0] == steps and ch_ms_lens.shape[0] == steps, (ch_depth.shape[0], ch_cv.shape[0], ch_ms_lens.shape[0], steps)

    # Model
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator)
    generator.eval()

    if torch.cuda.is_available():
        generator.cuda()

    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    start_time = time.time()

    sim_muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True):
        # cond [num_samples, 6]
        if len(changes) > 0:
            cond = torch.vstack((
                num[:, sp],
                depth[:, sp] * ch_depth.loc[sp, :].values,
                angle[:, sp],
                iz[:, sp],
                cv[:, sp] * ch_cv.loc[sp, :].values,
                length[:, sp] * ch_ms_lens.loc[sp, :].values,
            )).transpose(1, 0)
        else:
            cond = torch.vstack((
                num[:, sp],
                depth[:, sp],
                angle[:, sp],
                iz[:, sp],
                cv[:, sp],
                length[:, sp],
            )).transpose(1, 0)

        if torch.cuda.is_available():
            cond = cond.cuda()

        if mode == 'morph':
            if torch.cuda.is_available():
                muaps = muaps.cuda()
            sim = generator.generate(muaps.unsqueeze(1), cond.float())
        elif mode == 'sample':
            if torch.cuda.is_available():
                zi = zi.cuda()
            sim = generator.sample(num_samples, cond.float(), cond.device, zi)

        # Remove .cpu() if you only use cpu
        sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        sim_muaps.append(sim)

    sim_muaps = np.array(sim_muaps)
    sim_muaps = np.transpose(sim_muaps, (1, 0, 2, 3, 4))
    print('--- %s seconds ---' % (time.time() - start_time))

    # Filtering, not required
    start_time = time.time()
    # low-pass filtering for smoothing
    fs = 4096.
    _, _, n_row, n_col, time_samples = sim_muaps.shape
    T = time_samples / fs
    cutoff = 800
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    order = 4
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    filtered_muaps = filtfilt(b, a, sim_muaps.reshape(-1, time_samples))
    sim_muaps = filtered_muaps.reshape(num_samples, steps, n_row, n_col, time_samples)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_muaps(sim_muaps, args.res_path)

    # Save data
    # np.save(os.path.join(args.res_path, 'muaps_{}.npy'.format(mode)), sim_muaps)
