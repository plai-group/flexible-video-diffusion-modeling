# Official repository for [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495)

![A 30 second clip from a video sampled on CARLA Town01.](https://www.cs.ubc.ca/~wsgh/fdm/video_arrays/carla-part-of-long-sample.gif)

This codebase is based off that of [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) with the modifications to create a video model as described in [Flexible Diffusion Modeling of Long Videos](https://arxiv.org/abs/2205.11495).

# Usage

Tested with Python 3.10 in a conda environment. We require the Python packages `mpi4py torch torchvision wandb blobfile tqdm moviepy imageio` as well as the command line tool `ffmpeg`. This repository itself should also be installed by running
```
pip install -e .
```

For sampling with the adaptive sampling schemes (adaptive-autoreg and adaptive-hierarchy-2), also install the `lpips` package with e.g. `pip install lpips`.

To run `scripts/video_fvd.py`, tensorflow and tensorflow\_hub are required. Install with e.g. `pip install tensorflow==2.8 tensorflow_hub==0.12.0`.

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

### Experimenting with different sampling schemes
Our sampling schemes are defined in `improved_diffusion/sampling_schemes.py`, including all those presented in the paper. To add a new one, create a new subclass of `SamplingSchemeBase` with a `next_indices` function (returning a pair of vectors of observed and latent indices) in this file. Add it to the `sampling_schemes` dictionary (also in the same file) to allow your sampling scheme to be accessed by `scripts/video_sample.py`.
  
For debugging, you can visualise the indices used by a sampling scheme with 
```
python scripts/video_sample.py <ANY CHECKPOINT PATH> --sampling_scheme <SAMPLING SCHEME> --just_visualise
```
The `--just_visualise` flag tells the script to save a .png visualisation to the `visualisations/` directory instead of sampling videos.
  
  
### Directory structures
Checkpoints are saved with the following directory structure
```
checkpoints
├── .../<wandb id>
│   ├── model_<step>.pt
│   ├── ema_<ema_rate>_<step>.pt
│   └── opt_<step>.pt
└── ... (other runs)
```
Optionally, to better organise the results directories, you can make more descriptive directory names and move the checkpoints into them as follows. The results directory (containing samples etc.) will mirror this structure.
```
checkpoints
├── <descriptive path>
|   ├── <wandb id>
│   |   └── ema_<ema_rate>_<step>.pt
|   └── ... (other runs with same descriptive path)
└── ... (other runs)
```
After running sampling scripts, a results directory will be created with structure:
```
results
├── <descriptive path>
│   ├── <wandb id>
│   │   ├── <checkpoint name>
│   │   │   ├── <sampling scheme descriptor>
│   │   │   │   ├── samples
│   │   │   │   │  ├── <name-1>.npy
│   │   │   │   │  ├── <name-2>.npy
│   │   │   │   │  ├── ...
|   └── ... (other runs with same descriptive path)
└── ... (other runs)
```
FVD scores and video files created by `scripts/video_fvd.py` and `scripts/video_make_mp4.py` will also be saved in the `results/<descriptive path>/<wandb id>/<checkpoint name>/<sampling scheme descriptor>` directory.
  
## Computing FVD scores
After drawing sufficiently many samples (we use 100 for results reported in our paper), FVD scores can be computed with
```
python scripts/video_fvd.py  --eval_dir results/<descriptive path>/<wandb id>/<checkpoint name>/<sampling scheme descriptor> --num_videos <NUM VIDEOS>
```
Running this script will print the FVD as well as saving it in a file at `results/<descriptive path>/<wandb id>/<checkpoint name>/<sampling scheme descriptor>`.

## View-able video formats
Videos produced by `scripts/video_sample.py` are saved in `.npy` format. Save a video (or grid of sampled videos) in a `.gif` or `.mp4` format with
```
python scripts/video_make_mp4.py --eval_dir results/<descriptive path>/<wandb id>/<checkpoint name>/<sampling scheme descriptor>
```

## Interpretable metrics on CARLA Town01
To compute the WD and PO metrics on CARLA Town01 video samples, download our regressor to world coordinates using
```
mkdir checkpoints/carla-regressor/
cd checkpoints/carla-regressor/
wget https://www.cs.ubc.ca/~wsgh/fdm/carla-regressor-ckpts/classifier_checkpoint.pth 
wget https://www.cs.ubc.ca/~wsgh/fdm/carla-regressor-ckpts/regressor_checkpoint.pth
cd ../../
```
The regressor and classifier are run in tandem so both are required. After downloading them, you can map generated samples to the corresponding sequences of CARLA world coordinates with
```
python scripts/video_to_world_coords.py --eval_dir results/<descriptive path>/<wandb id>/<checkpoint name>/<sampling scheme descriptor> --regressor_path checkpoints/carla-regressor/regressor_checkpoint.pth --classifier_path checkpoints/carla-regressor/classifier_checkpoint.pth
```
Similarly, you can obtain the regressor's output on the dataset with
```
python scripts/video_to_world_coords.py --dataset_dir datasets/carla/no-traffic/ --regressor_path checkpoints/carla-regressor/regressor_checkpoint.pth --classifier_path checkpoints/carla-regressor/classifier_checkpoint.pth
```

We recommend using the pretrained classifier and regressor weights linked to above. To instead retrain these models run 
```
python scripts/carla_regressor_train.py --data_dir datasets/carla/no-traffic --is_classifier True
```
for the classifier and
```
python scripts/carla_regressor_train.py --data_dir datasets/carla/no-traffic --is_classifier False
```
for the regressor.
  
## Link to original (pre-refactor) codebase
This is a refactored version of [our original codebase](https://github.com/wsgharvey/video-diffusion) with which the experiments in [the paper](https://arxiv.org/abs/2205.11495) were run. This refactored codebase is significantly cleaner and with less changes from the Improved DDPM repo it is based on, as well as having an architectural simplification vs [our original codebase](https://github.com/wsgharvey/video-diffusion) (we removed positional encodings). We have reproduced the main results with this refactored codebase. Much of the functionality which is listed in this repository as being committed by [wsgharvey](https://github.com/wsgharvey/) was originally written by [saeidnp](https://github.com/saeidnp/) or [vmasrani](https://github.com/vmasrani).
