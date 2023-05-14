import torch
import joblib
import random
import numpy as np
from torch.utils import data
from tqdm import tqdm

import sys
sys.path.append('.')
from utils.params import TESTSET1, TRAINSET1, w_amp, coeff_a_a, coeff_a_b, coeff_cv_a, coeff_cv_b, coeff_fb_a, coeff_fb_b, coeff_iz_a, coeff_iz_b, coeff_len_a, coeff_len_b, coeff_r_a, coeff_r_b


class MuapWave(data.Dataset):
    """
    Simulated MUAPs from a realistic forearm finite element-based model
    Replace the real_path with your own file paths
    """

    def __init__(self, cfg, real_path='../muap_data/'):
        super(MuapWave, self).__init__()

        self.data_type = cfg.Type
        self.cfg = cfg

        if self.data_type == 'train_real':
            self._items1 = self._load_pt(real_path, 'train', 1)
            self._items2 = self._load_pt(real_path, 'train', 2)
            self._num_items = len(self._items1['iz']) + len(self._items2['iz'])

        elif self.data_type == 'test_real':
            self._items1 = self._load_pt(real_path, 'test', 1)
            self._items2 = self._load_pt(real_path, 'test', 2)
            self._num_items = len(self._items1['iz']) + len(self._items2['iz'])

        if self.data_type == 'test_real':
            self.tgt_idx = np.random.permutation(self._num_items)

    def __getitem__(self, index):
        if self.data_type == 'train_real':
            id1, id2 = random.sample(range(self._num_items), 2)
            tgt = self._select_item('train', id1)
            sp = self._select_item('train', id2)
            src = self._select_item('train', index)
            return src, tgt, sp

        elif self.data_type == 'test_real':
            tgt_idx = self.tgt_idx[index]
            src = self._select_item('test', index)
            tgt = self._select_item('test', tgt_idx)
            return src, tgt

    def __len__(self):
        return self._num_items

    def _load_pt(self, data_path, mode, id):
        db = joblib.load(data_path + mode + '_dataset' + str(id) + '.pt', 'r')
        return db

    def _select_item(self, mode, index):
        if mode == 'train':
            if index < TRAINSET1:
                db = self._items1
                idx = index
            else:
                db = self._items2
                idx = index - TRAINSET1
        elif mode == 'test':
            if index < TESTSET1:
                db = self._items1
                idx = index
            else:
                db = self._items2
                idx = index - TESTSET1

        ele = {
            'depth': (db['mu_depth'][idx].copy() + coeff_r_a) * coeff_r_b,
            'num_fibre_log': (db['num_fiber_log'][idx].copy() + coeff_fb_a) * coeff_fb_b,
            'angle': (db['mu_angle'][idx].copy() + coeff_a_a) * coeff_a_b,
            'iz': (db['iz'][idx].copy() + coeff_iz_a) * coeff_iz_b,
            'cv': (db['velocity'][idx].copy() + coeff_cv_a) * coeff_cv_b,
            'len': (db['len'][idx].copy() + coeff_len_a) * coeff_len_b,
            'hd_wave': torch.from_numpy(np.array(db['hd_wave'][idx])).float() * w_amp,
        }
        return ele


if __name__ == '__main__':
    dataset = MuapWave(data_type='train_real', num=3)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=32, shuffle=False, num_workers=1)
    data_bar = tqdm(dataloader, desc='Test MuapWave Class', dynamic_ncols=True)
    for src, tgt, sp in data_bar:
        src_muap = src['hd_wave'].permute(0, 3, 1, 2)

        print('src_muap.shape: ', src_muap.shape)
        print(tgt['iz'].shape)
        print(sp['hd_wave'].shape)

        assert(1 == 2)
