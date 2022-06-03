# Official repository for [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)

![A 30 second clip from a video sampled on CARLA Town01.](https://www.cs.ubc.ca/~wsgh/fdm/video_arrays/carla-part-of-long-sample.gif)

This codebase is based off that of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) with the modifications to create a video model as described in [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495).

# Usage

Tested with Python 3.10 in a conda environment. Requires `ffmpeg`. Install Python requirements as follows (run from inside git repo).
```
conda install -c conda-forge mpi4py
pip install torch torchvision wandb blobfile tqdm moviepy imageio
pip install -e .
```

This repo logs to wandb, using the wandb entity/username and project name set by:
```
export WANDB_ENTITY=<...>
export WANDB_PROJECT=<...>
```

## Preparing Data
TODO

## Training
Testing with:
```
python scripts/video_train.py --dataset mazes_cwvae --batch_size 3 --diffusion_steps 32 --num_res_blocks 1 --num_channels 32 --max_frames 4 --ema_rate 0.5 --sample_interval 50
```
After training for >`save_interval` iterations, we can kill and resume training from the latest job with
```
python scripts/video_train.py --dataset mazes_cwvae --batch_size 3 --diffusion_steps 32 --num_res_blocks 1 --num_channels 32 --max_frames 4 --ema_rate 0.5 --sample_interval 50 --wandb_id <WANDB ID OF RUN WE ARE RESUMING>
```
