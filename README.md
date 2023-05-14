# BioMime
Pytorch implementation and pretrained models for BioMime. For details, see [**Human Biophysics as Network Weights: Conditional Generative Models for Dynamic Simulation**](https://arxiv.org/abs/2211.01856)


## Requirements
- Operating System: Linux.
- Python 3.7.11
- PyTorch >= 1.6
- torchvision >= 0.8.0
- CUDA toolkit 10.1 or newer, cuDNN 7.6.3 or newer.


### Conda environment
environment.yml contains all the dependencies required to run BioMime. Create the new environment by:

```bash
conda env create --file environment.yml
```


## Data for training
Please contact [neurodec](http://neurodec.ai/) for the dataset.


## Pretrained models
Download [model.pth](https://drive.google.com/drive/folders/17Z2QH5NNaIv9p4iDq8HqytFaYk9Qnv2C?usp=sharing) and put them under `ckp/.`


## Quick Start
### Train
When you have your data ready, please follow the instructions below to train your own BioMime:
1. Edit utils/data.py to specify the path for dataset.
2. Configure the models and setting up in config/config.yaml.
3. Run the training script by:

```bash
python train.py --exp=test
```
Define your own experiment id by changing the argument `--exp`.

### Test
The checkpoints at snapshot epochs will be saved in res/exp/. You can test the model by:

```bash
python test.py --ckp_pth=./ckp/linear_anneal.pth --num_sample=32 --plot=1
```

### Generate
You can generate your own MUAPs by sampling from the standard Normal Distribution:
```bash
python generate.py --cfg config.yaml --mode sample --model_pth ./ckp/model_linear.pth --res_path ./res
```
Or by morphing the existing MUAPs:
```bash
python generate.py --cfg config.yaml --mode morph --model_pth ./ckp/model_linear.pth --res_path ./res
```
Make sure you have the file containing MUAPs in the format of [num, nrow, ncol, ntime] and set the argument `--data_path`. Examples of MUAP files will be provided in the future.

We also allow users to generate dynamic MUAPs during a realistic forearm movement defined by a musculoskeletal model. This new function will be available soon.


## Licenses
This repository is released under the GNU General Public License v3.0.


## Citation
```
@article{ma2022human,
  title={Human Biophysics as Network Weights: Conditional Generative Models for Ultra-fast Simulation},
  author={Ma, Shihan and Clarke, Alexander Kenneth and Maksymenko, Kostiantyn and Deslauriers-Gauthier, Samuel and Sheng, Xinjun and Zhu, Xiangyang and Farina, Dario},
  journal={arXiv preprint arXiv:2211.01856},
  year={2022}
}
```
