import sys

from utils.loss_functions import nrmse_matrix_torch
sys.path.append('.')
import torch
from tqdm import tqdm
import numpy as np

from utils.args import args, cfg
from utils.data import MuapWave

from MorphSys.generator import Generator


if __name__ == '__main__':
    BATCH_SIZE = cfg.Dataset.Batch

    # Dataset
    if cfg.Dataset.Type == 'MuapWave':
        test_dataset = MuapWave(cfg.Dataset.Test, subset=args.test_subset)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=cfg.Dataset.num_workers, pin_memory=True)

    # Model
    generator = Generator(cfg.Dataset.num_cond, cfg.Model.Generator.Latent)

    dest_path = './exp/{}_{}'.format(cfg.Dataset.Type, args.exp)
    checkpoint = torch.load(dest_path + '/epoch-{}.pth'.format(args.epoch_name))
    generator.load_state_dict(checkpoint['generator'])

    if torch.cuda.is_available():
        generator.cuda()

    generator.eval()

    test_bar = tqdm(test_data_loader, desc='Test MorphSys...', dynamic_ncols=True)

    sp_num = 0
    sim_rmse = np.zeros(args.num_sample)
    sp_rmse = np.zeros(args.num_sample)
    rev_rmse = np.zeros(args.num_sample)

    for test_src, test_tgt in test_bar:
        sp_num += BATCH_SIZE
        if sp_num > args.num_sample:
            break
        test_src_muap = test_src['hd_wave'].permute(0, 3, 1, 2)
        test_tgt_muap = test_tgt['hd_wave'].permute(0, 3, 1, 2)

        cond_src = torch.stack((
            test_src['num_fibre_log'],
            test_src['depth'],
            test_src['angle'],
            test_src['iz'],
            test_src['cv'],
            test_src['len']
        ), dim=1)
        cond_tgt = torch.stack((
            test_tgt['num_fibre_log'],
            test_tgt['depth'],
            test_tgt['angle'],
            test_tgt['iz'],
            test_tgt['cv'],
            test_tgt['len']
        ), dim=1)

        if torch.cuda.is_available():
            test_src_muap, cond_src, test_tgt_muap, cond_tgt = test_src_muap.cuda(), cond_src.cuda(), test_tgt_muap.cuda(), cond_tgt.cuda()

        sim = generator.generate(test_src_muap.unsqueeze(1), cond_tgt.float())
        sample = generator.sample(BATCH_SIZE, cond_tgt.float(), sim.device)
        rev = generator.generate(sim.unsqueeze(1), cond_src.float())

        sim_rmse[sp_num - BATCH_SIZE: sp_num] = nrmse_matrix_torch(sim, test_tgt_muap).cpu().detach().numpy()
        sp_rmse[sp_num - BATCH_SIZE: sp_num] = nrmse_matrix_torch(sample, test_tgt_muap).cpu().detach().numpy()
        rev_rmse[sp_num - BATCH_SIZE: sp_num] = nrmse_matrix_torch(rev, test_src_muap).cpu().detach().numpy()

    print('Averaged nRMSE between sim/gt: {:.2f}, sp/gt: {:.2f}, rev/src: {:.2f}, number of datapoints: {}'.format(sim_rmse.mean() * 100, sp_rmse.mean() * 100, rev_rmse.mean() * 100, args.num_sample))
