"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import json

import numpy as np
import torch

from improved_diffusion import dist_util
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    str2bool,
)
from improved_diffusion.test_util import get_model_results_path, get_eval_run_identifier, Protect
from improved_diffusion.sampling_schemes import sampling_schemes
from improved_diffusion.video_datasets import get_test_dataset


@torch.no_grad()
def infer_video(args, model, diffusion, batch):
    """
    batch has a shape of BxTxCxHxW where
    B: batch size
    T: video length
    CxWxH: image size
    """
    B, T, C, H, W = batch.shape
    samples = torch.zeros_like(batch)
    samples[:, :args.n_obs] = batch[:, :args.n_obs]
    # Intilise sampling scheme
    optimal_schedule_path = None if args.optimality is None else args.eval_dir / "optimal_schedule.pt"
    frame_indices_iterator = iter(sampling_schemes[args.sampling_scheme](
        video_length=T, num_obs=args.n_obs,
        max_frames=args.max_frames, step_size=args.max_latent_frames,
        optimal_schedule_path=optimal_schedule_path,
    ))

    while True:
        frame_indices_iterator.set_videos(samples.to(args.device))  # ignored for non-adaptive sampling schemes
        try:
            obs_frame_indices, latent_frame_indices = next(frame_indices_iterator)
        except StopIteration:
            break
        print(f"Conditioning on {sorted(obs_frame_indices)} frames, predicting {sorted(latent_frame_indices)}.\n")
        # Prepare network's input
        frame_indices = torch.cat([torch.tensor(obs_frame_indices), torch.tensor(latent_frame_indices)], dim=1)
        x0 = torch.stack([samples[i, fi] for i, fi in enumerate(frame_indices)], dim=0).clone()
        obs_mask = torch.cat([torch.ones_like(torch.tensor(obs_frame_indices)),
                              torch.zeros_like(torch.tensor(latent_frame_indices))], dim=1).view(B, -1, 1, 1, 1).float()
        latent_mask = 1 - obs_mask
        # Prepare masks
        print(f"{'Frame indices':20}: {frame_indices[0].cpu().numpy()}.")
        print(f"{'Observation mask':20}: {obs_mask[0].cpu().int().numpy().squeeze()}")
        print(f"{'Latent mask':20}: {latent_mask[0].cpu().int().numpy().squeeze()}")
        print("-" * 40)
        # Move tensors to the correct device
        x0, obs_mask, latent_mask, frame_indices = (t.to(batch.device) for t in [x0, obs_mask, latent_mask, frame_indices])
        # Run the network
        local_samples, _ = diffusion.p_sample_loop(
            model, x0.shape, clip_denoised=args.clip_denoised,
            model_kwargs=dict(frame_indices=frame_indices,
                              x0=x0,
                              obs_mask=obs_mask,
                              latent_mask=latent_mask),
            latent_mask=latent_mask,
            return_attn_weights=False)
        # Fill in the generated frames
        for i, li in enumerate(latent_frame_indices):
            samples[i, li] = local_samples[i, -len(li):].cpu()
    return samples


def main(args, model, diffusion, dataset):
    not_done = list(args.indices)
    while len(not_done) > 0:
        batch_indices = not_done[:args.batch_size]
        not_done = not_done[args.batch_size:]
        output_filenames = [args.eval_dir / "samples" / f"sample_{i:04d}-{args.sample_idx}.npy" for i in batch_indices]
        todo = [not p.exists() for p in output_filenames]
        if not any(todo):
            print(f"Nothing to do for the batches {min(batch_indices)} - {max(batch_indices)}, sample #{args.sample_idx}.")
            continue
        batch = torch.stack([dataset[i][0] for i in batch_indices])
        samples = infer_video(args, model, diffusion, batch)
        drange = [-1, 1]
        samples = (samples.numpy() - drange[0]) / (drange[1] - drange[0]) * 255
        samples = samples.astype(np.uint8)
        for i in range(len(batch_indices)):
            if todo[i]:
                np.save(output_filenames[i], samples[i])
                print(f"*** Saved {output_filenames[i]} ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--sampling_scheme", required=True, choices=sampling_schemes.keys())
    parser.add_argument("--n_obs", type=int, default=36, help="Number of observed frames at the beginning of the video. The rest are sampled.")
    parser.add_argument("--T", type=int, default=None, help="Length of the videos. If not specified, it will be inferred from the dataset.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Denoted K in the paper. Maximum number of (observed or latent) frames input to the model at once. Defaults to what the model was trained with.")
    parser.add_argument("--max_latent_frames", type=int, default=None, help="Number of frames to sample in each stage. Defaults to max_frames/2.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--stop_index", type=int, default=None)
    parser.add_argument("--use_ddim", type=str2bool, default=False)
    parser.add_argument("--timestep_respacing", type=str, default="")
    parser.add_argument("--clip_denoised", type=str2bool, default=True)
    parser.add_argument("--sample_idx", type=int, default=0, help="Sampled images will have this specific index. Used for sampling multiple videos with the same observations.")
    parser.add_argument("--just_visualise", action='store_true', help="Make visualisation of sampling scheme instead of doing it.")
    parser.add_argument("--optimality", type=str, default=None,
                        choices=["linspace-t", "random-t", "linspace-t-force-nearby", "random-t-force-nearby"],
                        help="Type of optimised sampling scheme to use for choosing observed frames. By default uses non-optimized sampling scheme. The optimal indices should be computed before use via video_optimal_schedule.py.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Prepare which indices to sample (for unconditional generation index does nothing except change file name)
    if args.stop_index is None:
        # assume we're in a slurm batch job, set start and stop_index accordingly
        assert "SLURM_ARRAY_TASK_ID" in os.environ
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        args.start_index = task_id * args.batch_size
        args.stop_index = (task_id + 1) * args.batch_size
    args.indices = list(range(args.start_index, args.stop_index))
    print(f"Sampling for indices {args.start_index} to {args.stop_index}.")

    # Load the checkpoint (state dictionary and config)
    data = dist_util.load_state_dict(args.checkpoint_path, map_location="cpu")
    state_dict = data["state_dict"]
    model_args = data["config"]
    model_args.update({"use_ddim": args.use_ddim,
                       "timestep_respacing": args.timestep_respacing})
    model_args = argparse.Namespace(**model_args)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(model_args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    args.image_size = model_args.image_size
    if args.max_frames is None:
        args.max_frames = model_args.max_frames
    if args.max_latent_frames is None:
        args.max_latent_frames = args.max_frames // 2 


    # Load the dataset (to get observations from)
    dataset = get_test_dataset(dataset_name=model_args.dataset, T=args.T)
    args.T = dataset.T

    if args.just_visualise:
        visualise(args)  # TODO
        exit()

    # Prepare samples directory
    args.eval_dir = get_model_results_path(args) / get_eval_run_identifier(args)
    (args.eval_dir / 'samples').mkdir(parents=True, exist_ok=True)
    print(f"Saving samples to {args.eval_dir / 'samples'}")

    # Store model configs in a JSON file
    json_path = args.eval_dir / "model_config.json"
    if not json_path.exists():
        with Protect(json_path): # avoids race conditions
            with open(json_path, "w") as f:
                json.dump(vars(model_args), f, indent=4)
        print(f"Saved model config at {json_path}")

    main(args, model, diffusion, dataset)