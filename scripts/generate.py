import argparse
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, filtfilt

from BioMime.utils.basics import update_config, load_generator
from BioMime.utils.plot_functions import plot_muaps
from BioMime.utils.params import num_mus, steps, tgt_params
from BioMime.models.generator import Generator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate MUAPs')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')

    parser.add_argument('--model_pth', required=True, type=str, help='file of best model')
    parser.add_argument('--mode', default='sample', type=str, help='sample or morph')
    parser.add_argument('--data_path', default='default', type=str, help='file of data to morph')
    parser.add_argument('--res_path', required=True, type=str, help='path of result folder')
    parser.add_argument('--device', default='cuda', type=str, help='cuda|cpu')

    args = parser.parse_args()
    cfg = update_config('./config/' + args.cfg)

    # Check modes
    mode = args.mode
    if mode == 'morph':
        assert args.data_path != 'default', 'Datapath for existing MUAPs required.'
        muaps = np.load(args.data_path)
        num_mus = muaps.shape[0]
    else:
        zi = torch.randn(num_mus, cfg.Model.Generator.Latent)

    # Model
    generator = Generator(cfg.Model.Generator)
    generator = load_generator(args.model_pth, generator, args.device)
    generator.eval()

    # Device
    if args.device == 'cuda':
        assert torch.cuda.is_available()

    if args.device == 'cuda':
        generator.cuda()

    if not os.path.exists(args.res_path):
        os.mkdir(args.res_path)

    start_time = time.time()

    sim_muaps = []
    for sp in tqdm(range(steps), dynamic_ncols=True):
        # cond [num_mus, 6]
        cond = torch.vstack((
            tgt_params['num'][:, sp],
            tgt_params['depth'][:, sp],
            tgt_params['angle'][:, sp],
            tgt_params['iz'][:, sp],
            tgt_params['cv'][:, sp],
            tgt_params['length'][:, sp],
        )).transpose(1, 0)

        if args.device == 'cuda':
            cond = cond.cuda()

        if mode == 'morph':
            if args.device == 'cuda':
                muaps = muaps.cuda()
            sim = generator.generate(muaps.unsqueeze(1), cond.float())
        elif mode == 'sample':
            if args.device == 'cuda':
                zi = zi.cuda()
            sim = generator.sample(num_mus, cond.float(), cond.device, zi)

        if args.device == 'cuda':
            sim = sim.permute(0, 2, 3, 1).cpu().detach().numpy()
        else:
            sim = sim.permute(0, 2, 3, 1).detach().numpy()
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
    sim_muaps = filtered_muaps.reshape(num_mus, steps, n_row, n_col, time_samples)
    print("--- %s seconds ---" % (time.time() - start_time))

    plot_muaps(sim_muaps, args.res_path)

    # Save data
    np.save(os.path.join(args.res_path, 'muaps_{}.npy'.format(mode)), sim_muaps)
