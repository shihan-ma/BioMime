import argparse
import sys
sys.path.append('.')

from utils.basics import update_config

# Define arguments
parser = argparse.ArgumentParser(description='Morph Waveforms')
parser.add_argument('--cfg', type=str, default='config.yaml', help='Name of configuration file')

# Experiment settings
parser.add_argument('--exp', default='default', type=str, help='Experiment id')
parser.add_argument('--exp_load', default='default', type=str, help='Experiment id that\'s to be loaded')
parser.add_argument('--load_ckp', default=-1, type=int, help='Epoch of the experiment that\'s to be loaded')

# Testing
parser.add_argument('--ckp_path', type=str, default='./ckp/model_linear.pth', help='Pretrained checkpoint')
parser.add_argument('--num_sample', type=int, default=96000, help='Number of samples for testing')
parser.add_argument('--plot', type=int, default=0, help='Plot the final batch - 1 or not - 0 for testing dataset')

args = parser.parse_args()
cfg = update_config('./config/' + args.cfg)

print(cfg)
