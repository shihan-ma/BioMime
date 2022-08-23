# MorphSys
Deep latent generative model that learns human biophysics.


## Paper (Preprint)


## Requirements
- Operating System: Linux.
- Python 3.7.11
- PyTorch >= 1.6
- torchvision >= 0.8.0
- CUDA toolkit 10.1 or newer, cuDNN 7.6.3 or newer.


### Conda environment
environment.yml contains all the dependencies required to run MorphSys. Create the new environment by:

```bash
conda env create --file environment.yml
```


## Data for training
Please contact [neurodec](http://neurodec.ai/) for the dataset.

## Quick Start
### Train
When you have your data ready, please follow the instructions below to train your own MorphSys:
1. Edit utils/data.py to specify the path for dataset.
2. Configure the models and setting up in config/config.yaml.
3. Run the training script by:

```bash
python train_morphsys.py --exp=test
```
Define your own experiment id by changing the argument exp.

### Test
The checkpoints at snapshot epochs will be saved in res/exp/. You can test the model by:

```bash
python test_morphsys.py --exp=test --epoch_name=50 --num_sample=32 --plot=1
```


## Licenses


## Citation
