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

And add a directory for checkpoints to be saved in:
```
mkdir checkpoints
```

## Preparing Data
### CARLA Town01
Download CARLA Town01 with our download script as follows.
```
cd datasets/carla
bash download.sh
cd ../..
```

### MineRL and GQN-Mazes
TODO

## Training
### Running on CARLA Town01
We train models on CARLA Town01 on an A100 GPU with the command:
```
python scripts/video_train.py --batch_size=2 --max_frames 20 --dataset=carla_no_traffic --num_res_blocks=1
```
### Debugging/running on smaller GPUs
To train on smaller GPUs without exceeding CUDA memory limits, you can try reducing `--max_frames` from e.g. 20 to 5, `--batch_size` from 2 to 1, and `--num_channels` from the default 128 to e.g. 32.

### Debugging with faster feedback loops
All of the previous suggestions for running on smaller GPUs will usually also speed up training. To more quickly check for major issues with training, you can decrease `--sample_interval` from the default 50000 to e.g. 1000 so that samples are logged to wandb more often, and decrease `--diffusion_steps` from the default 1000 to e.g. 32 so that logging samples is faster.

### Resuming after training ends/fails
After training for more than `save_interval` iterations (50000 by default), we can kill and resume training from the latest checkpoint with:
```
python scripts/video_train.py <ORIGINAL ARGUMENTS> --wandb_id <WANDB ID OF RUN WE ARE RESUMING>
```
e.g.
```
python scripts/video_train.py --batch_size=2 --max_frames 20 --dataset=carla_no_traffic --num_res_blocks=1 --wandb_id 1v1myd4c
```


## Sampling
Checkpoints are saved throughout training to paths of the form `checkpoints/<WANDB ID>/model<NUMBER OF ITERATIONS>.pt` and `checkpoints/<WANDB ID>/ema_<EMA RATE>_<NUMBER OF ITERATIONS>.pt` respectively. Best results can usually be obtained from the exponential moving averages (EMAs) of model weights saved in the latter form. Given a trained checkpoint, we can sample from it with a command like
```
python scripts/video_sample.py <CHECKPOINT PATH> --batch_size 2 --sampling_scheme <SAMPLING SCHEME> --stop_index <STOP INDEX> --n_obs <N OBS>
```
which will sample completions for the first <STOP INDEX> test videos, each conditioned on the first <N OBS> frames (where <N OBS> may be zero). The dataset to use and other hyperparameters are inferred from the specified checkpoint. The <SAMPLING SCHEME> should be one of those defined in `improved_diffusion/sampling_schemes.py`, most of which are described in the paper. Options include, "autoreg", "long-range", "hierarchy-2", "adaptive-autoreg", "adaptive-hierarchy-2". The final command will look something like:
```
python scripts/video_sample.py checkpoints/2f1gq6ud/ema_0.9999_550000.pt --batch_size 2 --sampling_scheme autoreg --stop_index 100
```
