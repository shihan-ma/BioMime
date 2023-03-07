"""
This script is used to prepare for the parameters that change during a specified movement.
Here we provide two examples:
1. simple linspace params
2. params changed with motions in msk models
"""

import torch
import numpy as np

from utils.pose_functions import pos2params

# define the number of motor units here
num_mus = 5

# # simple linspace params - all MUs in all muscles change consistently
# # note that the range is better within [0.5, 1.0]
# steps = 20
# num = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# depth = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# angle = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# iz = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# cv = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# length = torch.ones(num_mus, 1) * torch.linspace(0.6, 0.9, steps)
# changes = {}

# msk model - muscle-specific changes
# ----------- user defined -----------
fs = 5      # temporal frequency in Hz
poses = ['default', 'grasp', 'grasp+flex', 'grasp', 'grasp+ext', 'grasp', 'default']
durations = [2] * 6     # Note that len(durations) should be one less than len(poses) as it represents the intervals.
# ----------- user defined -----------

ms_labels = ['ECRL', 'ECRB', 'ECU', 'FCR', 'FCU', 'PL', 'FDSL', 'FDSR', 'FDSM', 'FDSI', 'FDPL', 'FDPR', 'FDPM', 'FDPI', 'EDCL', 'EDCR', 'EDCM', 'EDCI', 'EDM', 'EIP', 'EPL', 'EPB', 'FPL', 'APL', 'APB', 'FPB', 'OPP', 'ADPt', 'ADPo', 'ADM', 'FDM']
steps = fs * np.sum(durations)

ms_lens, depths, cvs = pos2params(poses, durations, ms_labels)
# define init absolute params, should be within [0.5, 1.0]
num = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5
# num = torch.linspace(0.5, 1.0, num_mus).repeat(1, steps)      # increasing fibre numbers
depth = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5
angle = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5
iz = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5
cv = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5
length = torch.rand(num_mus, 1).repeat(1, steps) / 2 + 0.5

changes = {'depth': depths, 'cv': cvs, 'len': ms_lens}
