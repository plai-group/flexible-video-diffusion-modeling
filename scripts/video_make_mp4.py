import torch
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import uuid
import argparse
import json

from improved_diffusion.video_datasets import get_test_dataset
from improved_diffusion.test_util import mark_as_observed, tensor2gif, tensor2mp4
from improved_diffusion.script_util import str2bool


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--add_gt", type=str2bool, default=True)
    parser.add_argument("--do_n", type=int, default=1)
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--obs_length", type=int, default=0,
                        help="Number of observed images. If positive, marks the first obs_length frames in output gifs by a red border.")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4", "avi"])
    args = parser.parse_args()

    if args.add_gt:
        model_args_path = Path(args.eval_dir) / "model_config.json"
        with open(model_args_path, "r") as f:
            model_args = argparse.Namespace(**json.load(f))
        dataset = get_test_dataset(model_args.dataset)
    out_dir = (Path(args.out_dir) if args.out_dir is not None else Path(args.eval_dir)) / "videos"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.do_n}_{args.n_seeds}.{args.format}"

    videos = []
    for data_idx in range(args.do_n):
        if args.add_gt:

            gt_drange = [-1, 1]
            gt_video, _ = dataset[data_idx]
            gt_video = (gt_video.numpy() - gt_drange[0]) / (gt_drange[1] - gt_drange[0])  * 255
            gt_video = gt_video.astype(np.uint8)
            mark_as_observed(gt_video)
            videos.append([gt_video])
        else:
            videos.append([])
        seed = 0
        done = 0
        while done < args.n_seeds:
            filename = Path(args.eval_dir) / "samples" / f"sample_{data_idx:04d}-{seed}.npy"
            print(filename)
            try:
                video = np.load(filename)
                mark_as_observed(video[:args.obs_length])
                videos[-1].append(video)
                done += 1
            except PermissionError:
                pass
            seed += 1
            assert seed < 100, f'Not enough seeds for idx {data_idx} (found {done} after trying {seed} seeds)'
        videos[-1] = np.concatenate(videos[-1], axis=-2)
    video = np.concatenate(videos, axis=-1)

    random_str = uuid.uuid4()
    if args.format == "gif":
        tensor2gif(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    elif args.format == "mp4":
        print(torch.tensor(video).shape, torch.tensor(video).dtype)
        tensor2mp4(torch.tensor(video), out_path, drange=[0, 255], random_str=random_str)
    else:
        raise ValueError(f"Unknown format {args.format}")
    print(f"Saved to {out_path}")