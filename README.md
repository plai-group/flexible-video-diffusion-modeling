# Official repository for [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)

![A 30 second clip from a video sampled on CARLA Town01.](https://www.cs.ubc.ca/~wsgh/fdm/video_arrays/carla-part-of-long-sample.gif)

This codebase is based off that of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) with the modifications to create a video model as described in [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495).

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
