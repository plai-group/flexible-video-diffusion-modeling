import os
from filelock import FileLock
from pathlib import Path
import torch as th
import numpy as np
from PIL import Image
import imageio


class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)


def get_model_results_path(args):
    """
        Given arguments passed to an evaluation run, returns the path to the results path.
        The path has the format "results/<checkpoint_dir_subpath>/checkpoint name" where
        <checkpoint_dir_subpath> is the subset of checkpoint path after ".*checkpoint.*/"
        For example, if "/scratch/video-diffusion/saeids-checkpoints/abcdefg/ema_latest.pt"
        is the checkpoint path, the result path will be
        "results/abcdefg/ema_latest_<checkpoint_step>/". In this path, <checkpoint_step> is the
        training step of the checkpoint and will only be added if the checkpoint path ends with
        "latest", since otherwise the checkpoint name itself ends with the step number.
        If args.eval_dir is not None, this function does nothing and returns the same path.
        args is expected to have the following attributes:
        - use_ddim
        - timesptep_respacing
        - outdir
    """
    # Extract the diffusion sampling arguments string (DDIM/respacing)
    postfix = ""
    if args.use_ddim:
        postfix += "_ddim"
    if args.timestep_respacing != "":
        postfix += "_" + f"respace{args.timestep_respacing}"

    # Create the output directory (if does not exist)
    if args.eval_dir is None:
        checkpoint_path = Path(args.checkpoint_path)
        name = f"{checkpoint_path.stem}"
        if name.endswith("latest"):
            checkpoint_step = th.load(args.checkpoint_path, map_location="cpu")["step"]
            name += f"_{checkpoint_step}"
        if postfix != "":
            name += postfix
        path = None
        for idx, x in enumerate(checkpoint_path.parts):
            if "checkpoint" in x:
                path = Path(*(checkpoint_path.parts[idx+1:]))
                break
        assert path is not None
        return Path("results") / path.parent / name
    else:
        return Path(args.eval_dir)


def get_eval_run_identifier(args):
    res = args.sampling_scheme
    if hasattr(args, "optimality") and args.optimality is not None:
        res += f"_optimal-{args.optimality}"
    res += f"_{args.max_frames}_{args.max_latent_frames}_{args.T}_{args.n_obs}"
    if hasattr(args, "dataset_partition") and args.dataset_partition == "train":
        res = "trainset_" + res
    return res


################################################################################
#                           Visualization functions                            #
################################################################################
def mark_as_observed(images, color=[255, 0, 0]):
    for i, c in enumerate(color):
        images[..., i, :, 1:2] = c
        images[..., i, 1:2, :] = c
        images[..., i, :, -2:-1] = c
        images[..., i, -2:-1, :] = c


def tensor2pil(tensor, drange=[0,1]):
    """Given a tensor of shape (Bx)3xwxh with pixel values in drange, returns a PIL image
       of the tensor. Returns a list of images if the input tensor is a batch.
    Args:
        tensor: A tensor of shape (Bx)3xwxh
        drange (list, optional): Range of pixel values in the input tensor. Defaults to [0,1].
    """
    assert tensor.ndim == 3 or tensor.ndim == 4
    if tensor.ndim == 3:
        return tensor2pil(tensor.unsqueeze(0), drange=drange)[0]
    img_batch = tensor.cpu().numpy().transpose([0, 2, 3, 1])
    img_batch = (img_batch - drange[0]) / (drange[1] - drange[0])  * 255 # img_batch with pixel values in [0, 255]
    img_batch = img_batch.astype(np.uint8)
    return [Image.fromarray(img) for img in img_batch]

def tensor2gif(tensor, path, drange=[0, 1], random_str=""):
    frames = tensor2pil(tensor, drange=drange)
    tmp_path = f"/tmp/tmp_{random_str}.png"
    res = []
    for frame in frames:
        frame.save(tmp_path)
        res.append(imageio.imread(tmp_path))
    imageio.mimsave(path, res)

def tensor2mp4(tensor, path, drange=[0, 1], random_str=""):
    gif_path = f"/tmp/tmp_{random_str}.gif"
    tensor2gif(tensor, path=gif_path, drange=drange, random_str=random_str)
    os.system(f"ffmpeg -y -hide_banner -loglevel error -i {gif_path} -r 10 -movflags faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" {path}")