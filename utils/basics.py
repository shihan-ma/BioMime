import os
import torch
import random
import numpy as np
import yaml
from easydict import EasyDict as edict


def save_model(models, optimizers, epoch, dataset, exp_id):
    """
    models: [generator, discriminator]
    """
    save_dict = {
        'epoch': epoch,
        'generator': models[0].state_dict(),
        'discriminator': models[1].state_dict(),
        'g_optimizer': optimizers[0].state_dict(),
        'd_optimizer': optimizers[1].state_dict(),
    }

    filename = os.path.join('./exp/{}_{}/epoch-{}_checkpoint.pth'.format(dataset, exp_id, epoch + 1))
    torch.save(save_dict, filename)


def load_model(epoch, dataset, exp_id, generator, discriminator, g_optimizer, d_optimizer):
    ckp_fpath = os.path.join('./exp/{}_{}/epoch-{}_checkpoint.pth'.format(dataset, exp_id, epoch + 1))
    checkpoint = torch.load(ckp_fpath)
    generator.load_state_dict(checkpoint['generator'])

    discriminator.load_state_dict(checkpoint['discriminator'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer'])

    return generator, discriminator, g_optimizer, d_optimizer, checkpoint['epoch']


def load_generator(ckp_fpath, generator, device='cuda'):
    checkpoint = torch.load(ckp_fpath, map_location=torch.device(device))
    generator.load_state_dict(checkpoint)

    return generator


# Set random seed for reproducibility.
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


class DataLogger(object):
    """Average data logger."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def kl_anneal_function(anneal_function, step, k=1, x0=30000):
    if anneal_function == 'logistic':
        x = np.linspace(-6, 6, x0)
        y = 1 / (1 + np.exp(-k * (x)))
        if step >= x0:
            return 1
        else:
            return y[step]
    elif anneal_function == 'linear':
        return min(1, step / x0)


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
