This is a fork of the codebase for [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). I am modifying it to be compatible with my workflow (mainly by integration with wandb). This README contains the commands I have used for testing. More comprehensive instructions are available at [the original repo](https://github.com/openai/improved-diffusion).

# Usage

Tested with Python 3.8.13 in a conda environment. Installed requirements with:
```
# conda install pytorch torchvision -c pytorch  # I ran this line but it is probably unnecessary given that I reinstalled with pip
conda install -c conda-forge mpi4py
pip install wandb blobfile tqdm
pip install --upgrade torch torchvision
pip install -e .  # Run from inside this git repo. This installs the `improved_diffusion` python package that the scripts use
```

This repo logs to wandb, using the wandb entity/username and project name set by:
```
export WANDB_ENTITY=<...>
export WANDB_PROJECT=<...>
```

## Preparing Data
I only tested using CIFAR-10, preparing data with
```
cd datasets
python cifar10.py
cd ..
```

## Training
I tested with
```
python scripts/image_train.py --data_dir datasets/cifar_train/ --lr 1e-4 --batch_size 8 --diffusion_steps 32 --noise_schedule linear --image_size 64 --num_channels 64 --num_res_blocks 1 --save_interval 50
```
After training for >50 iterations, we can kill and resume training from the latest job with
```
python scripts/image_train.py --data_dir datasets/cifar_train/ --lr 1e-4 --batch_size 8 --diffusion_steps 32 --noise_schedule linear --image_size 64 --num_channels 64 --num_res_blocks 1 --save_interval 50 --wandb_id <WANDB ID OF RUN WE ARE RESUMING>
```
