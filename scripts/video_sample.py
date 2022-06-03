"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
from tkinter import N

import numpy as np
import torch as th

from improved_diffusion import dist_util  # we do NOT support distributed sampling
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    str2bool,
)
from improved_diffusion.test_util import get_model_results_path


def main(model, args):

    print("sampling...")
    saved = 0
    while saved < len(args.indices):
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        for img in sample.cpu().numpy():
            drange = [-1, 1]  # Range of the generated samples' pixel values
            img = (img - drange[0]) / (drange[1] - drange[0])  * 255  # recon with pixel values in [0, 255]
            img = img.astype(np.uint8)
            fname = args.eval_dir / f"sample-{saved:06d}.npy"
            np.save(fname, img)
            saved += 1

    print("sampling complete")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="If not None, only generate videos for the specified indices. Used for handling parallelization.")
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--clip_denoised", type=str2bool, default=True)
    parser.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Prepare samples directory
    args.eval_dir = get_model_results_path(args) / "samples"
    args.eval_dir.mkdir(parents=True, exist_ok=True)

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_args.update({"use_ddim": args.use_ddim,
                       "timestep_respacing": args.timestep_respacing})
    model_args = argparse.Namespace(**model_args)
    # Load the model
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    args.image_size = model_args.image_size

    # Prepare which image indices to sample (for unconditional generation, index does nothing except change file name)
    if args.indices is None:
        assert "SLURM_ARRAY_TASK_ID" in os.environ
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        args.indices = list(range(task_id * args.batch_size, (task_id + 1) * args.batch_size))
        print(f"Only generating predictions for the batch #{task_id}.")

    main(model, args)
