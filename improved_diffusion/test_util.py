from filelock import FileLock
from pathlib import Path
import torch

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
            checkpoint_step = torch.load(args.checkpoint_path, map_location="cpu")["step"]
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