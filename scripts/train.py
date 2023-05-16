import os
import sys
sys.path.append('.')
import torch
from torch import optim
from tqdm import tqdm

from BioMime.models.discriminator import Discriminator
from BioMime.models.generator import Generator
from BioMime.utils.args import args, cfg
from BioMime.utils.data import MuapWave
from BioMime.utils.basics import kl_anneal_function, setup_seed, load_model, save_model, DataLogger
from BioMime.utils.loss_functions import nrmse_matrix_torch


if __name__ == '__main__':
    BATCH_SIZE = cfg.Dataset.Batch

    # Dataset
    if cfg.Dataset.Type == 'MuapWave':
        train_dataset = MuapWave(cfg.Dataset.Train)
        test_dataset = MuapWave(cfg.Dataset.Test)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cfg.Dataset.num_workers, pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=cfg.Dataset.num_workers, pin_memory=True)

    # Setting seed
    manualSeed = 3723
    setup_seed(manualSeed)

    # Model
    discriminator = Discriminator(cfg.Model.Discriminator)
    generator = Generator(cfg.Model.Generator)

    if torch.cuda.is_available():
        print('cuda available')
        discriminator.cuda()
        generator.cuda()

    print("# generator parameters:", sum(param.numel() for param in generator.parameters()))
    print("# discriminator parameters:", sum(param.numel() for param in discriminator.parameters()))

    # optimizers
    g_optimizer = optim.RMSprop(generator.parameters(), lr=cfg.Optimizer.G)
    d_optimizer = optim.RMSprop(discriminator.parameters(), lr=cfg.Optimizer.D)

    if args.load_ckp >= 0:
        print('Loading previously trained models')
        generator, discriminator, g_optimizer, d_optimizer, last_epoch = load_model(args.load_ckp, cfg.Dataset.Type, args.exp_load, generator, discriminator, g_optimizer, d_optimizer)
    else:
        last_epoch = -1

    start_epoch = last_epoch + 1

    num_iter = 0
    for epoch in range(start_epoch, cfg.Training.num_epoch):
        # TRAIN MODEL
        generator.train()
        discriminator.train()

        d_real_logger = DataLogger()
        d_sim_logger = DataLogger()
        g_gan_logger = DataLogger()
        g_kld_logger = DataLogger()
        g_cycle_logger = DataLogger()
        sim_rmse_logger = DataLogger()
        sp_rmse_logger = DataLogger()
        rev_rmse_logger = DataLogger()

        train_bar = tqdm(train_data_loader, dynamic_ncols=True)

        for train_src, train_tgt, train_sp in train_bar:
            train_src_muap = train_src['hd_wave'].permute(0, 3, 1, 2)
            train_tgt_muap = train_tgt['hd_wave'].permute(0, 3, 1, 2)
            train_sp_muap = train_sp['hd_wave'].permute(0, 3, 1, 2)

            src_cond = torch.stack((
                train_src['num_fibre_log'],
                train_src['depth'],
                train_src['angle'],
                train_src['iz'],
                train_src['cv'],
                train_src['len']
            ), dim=1)
            tgt_cond = torch.stack((
                train_tgt['num_fibre_log'],
                train_tgt['depth'],
                train_tgt['angle'],
                train_tgt['iz'],
                train_tgt['cv'],
                train_tgt['len']
            ), dim=1)
            sp_cond = torch.stack((
                train_sp['num_fibre_log'],
                train_sp['depth'],
                train_sp['angle'],
                train_sp['iz'],
                train_sp['cv'],
                train_sp['len']
            ), dim=1)

            if torch.cuda.is_available():
                train_src_muap, train_tgt_muap, train_sp_muap, src_cond, tgt_cond, sp_cond = train_src_muap.cuda(), train_tgt_muap.cuda(), train_sp_muap.cuda(), src_cond.cuda(), tgt_cond.cuda(), sp_cond.cuda()

            # Train discriminator to recognize real signal and real condition as TRUE
            discriminator.zero_grad()
            d_optimizer.zero_grad()

            out = discriminator(train_sp_muap.unsqueeze(1), sp_cond.float())
            out = out.sigmoid()
            d_weight_real = 0.1
            real_loss = d_weight_real * torch.mean((out - 1.0) ** 2)

            # Train discriminator to recognize (simulated signals + real conditions) or (real signals + random conditions) as FALSE
            sim = generator.generate(train_src_muap.unsqueeze(1), tgt_cond.float())
            random_cond = torch.rand_like(tgt_cond) * 0.5 + 0.5     # normalize to [0.5, 1]
            out1 = discriminator(sim.unsqueeze(1), tgt_cond.float()).sigmoid()
            out2 = discriminator(train_sp_muap.unsqueeze(1), random_cond.float()).sigmoid()
            d_weight_sim = 0.1
            sim_loss1 = torch.mean(out1 ** 2)
            sim_loss2 = torch.mean(out2 ** 2)
            sim_loss = d_weight_sim * (sim_loss1 + sim_loss2) * 0.5

            d_loss = real_loss + sim_loss
            d_loss.backward()
            d_optimizer.step()
            d_real_logger.update(real_loss.item(), BATCH_SIZE)
            d_sim_logger.update(sim_loss.item(), BATCH_SIZE)

            # Train generator to misguide discriminator
            generator.zero_grad()
            g_optimizer.zero_grad()

            g_sim, mu, log_var = generator(train_src_muap.unsqueeze(1), tgt_cond.float())
            g_sp = generator.sample(BATCH_SIZE, tgt_cond.float(), g_sim.device)
            rev = generator.generate(g_sim.unsqueeze(1), src_cond.float())

            out = discriminator(g_sim.unsqueeze(1), tgt_cond.float())
            out = out.sigmoid()

            g_gan_loss = torch.mean((out - 1.0) ** 2)
            g_cycle_loss = torch.mean((rev - train_src_muap) ** 2)
            g_kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

            if cfg.Training.anneal_kld:
                kld_anneal_w = kl_anneal_function(cfg.Training.anneal_type, num_iter, x0=cfg.Training.anneal_iters)
                num_iter += 1
            else:
                kld_anneal_w = 1

            g_loss = cfg.Loss.w_gan * g_gan_loss + cfg.Loss.w_kld * kld_anneal_w * g_kld_loss + cfg.Loss.w_cycle * g_cycle_loss
            g_loss.backward()
            g_optimizer.step()

            g_gan_logger.update(g_gan_loss.item(), BATCH_SIZE)
            g_kld_logger.update(g_kld_loss.item(), BATCH_SIZE)
            g_cycle_logger.update(g_cycle_loss.item(), BATCH_SIZE)

            sim_rmse = nrmse_matrix_torch(g_sim, train_tgt_muap).mean()
            sp_rmse = nrmse_matrix_torch(g_sp, train_tgt_muap).mean()
            rev_rmse = nrmse_matrix_torch(rev, train_src_muap).mean()

            sim_rmse_logger.update(sim_rmse.item(), BATCH_SIZE)
            sp_rmse_logger.update(sp_rmse.item(), BATCH_SIZE)
            rev_rmse_logger.update(rev_rmse.item(), BATCH_SIZE)

            train_bar.set_description(
                'Epoch {}: d_real {:.4f}, d_sim {:.4f}, g_gan {:.4f}, g_cycle {:.4f}, g_kld {:.4f} | sim {:.2f}, sp {:.2f}, rev {:.2f}'.format(
                    epoch + 1, d_real_logger.avg, d_sim_logger.avg, g_gan_logger.avg, g_cycle_logger.avg, g_kld_logger.avg, sim_rmse_logger.avg * 100, sp_rmse_logger.avg * 100, rev_rmse_logger.avg * 100
                )
            )

        # TEST model
        generator.eval()
        discriminator.eval()

        sim_rmse_logger = DataLogger()
        sp_rmse_logger = DataLogger()
        rev_rmse_logger = DataLogger()

        if (epoch + 1) % cfg.Eval.snap_shot == 0:
            with torch.no_grad():
                test_bar = tqdm(test_data_loader, desc='Test BioMime and save models', dynamic_ncols=True)
                for test_src, test_tgt in test_bar:
                    test_src_muap = test_src['hd_wave'].permute(0, 3, 1, 2)
                    test_tgt_muap = test_tgt['hd_wave'].permute(0, 3, 1, 2)
                    src_cond = torch.stack((
                        train_src['num_fibre_log'],
                        train_src['depth'],
                        train_src['angle'],
                        train_src['iz'],
                        train_src['cv'],
                        train_src['len']
                    ), dim=1)
                    tgt_cond = torch.stack((
                        train_tgt['num_fibre_log'],
                        train_tgt['depth'],
                        train_tgt['angle'],
                        train_tgt['iz'],
                        train_tgt['cv'],
                        train_tgt['len']
                    ), dim=1)

                    if torch.cuda.is_available():
                        test_src_muap, test_tgt_muap, src_cond, tgt_cond = test_src_muap.cuda(), test_tgt_muap.cuda(), src_cond.cuda(), tgt_cond.cuda()

                    sim = generator.generate(test_src_muap.unsqueeze(1), tgt_cond.float())
                    rev = generator.generate(sim.unsqueeze(1), src_cond.float())
                    sample = generator.sample(BATCH_SIZE, tgt_cond.float(), sim.device)

                    sim_rmse = nrmse_matrix_torch(sim, test_tgt_muap).mean()
                    sp_rmse = nrmse_matrix_torch(sample, test_tgt_muap).mean()
                    rev_rmse = nrmse_matrix_torch(rev, test_src_muap).mean()
                    sim_rmse_logger.update(sim_rmse.item(), BATCH_SIZE)
                    sp_rmse_logger.update(sp_rmse.item(), BATCH_SIZE)
                    rev_rmse_logger.update(rev_rmse.item(), BATCH_SIZE)

                    test_bar.set_description(
                        'Epoch {}: sim_rmse {:.2f}, sp_rmse {:.2f}, rev_rmse {:.2f}'.format(
                            epoch + 1, sim_rmse_logger.avg * 100, sp_rmse_logger.avg * 100, rev_rmse_logger.avg * 100
                        )
                    )

            file_path = './exp/{}_{}'.format(cfg.Dataset.Type, args.exp)
            if not os.path.exists(file_path):
                os.mkdir(file_path)

            save_model([generator, discriminator], [g_optimizer, d_optimizer], epoch, cfg.Dataset.Type, args.exp)
