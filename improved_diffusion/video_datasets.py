import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from pathlib import Path
import shutil
from mpi4py import MPI

from .test_util import Protect


video_data_paths_dict = {
    "minerl":       "datasets/minerl_navigate-torch",
    "mazes_cwvae":  "datasets/gqn_mazes-torch",
    "carla_no_traffic": "datasets/carla/no-traffic",
}

default_T_dict = {
    "minerl":       500,
    "mazes_cwvae":  300,
    "carla_no_traffic": 1000,
}

default_image_size_dict = {
    "minerl":       64,
    "mazes_cwvae":  64,
    "carla_no_traffic": 128,
}


def load_data(dataset_name, batch_size, T=None, deterministic=False, num_workers=1, return_dataset=False):
    data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    if dataset_name == "minerl":
        data_path = os.path.join(data_path, "train")
        dataset = MineRLDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "train")
        dataset = GQNMazesDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "carla_no_traffic":
        dataset = CarlaDataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T)
    else:
        raise Exception("no dataset", dataset_name)
    if return_dataset:
        return dataset
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=num_workers, drop_last=True
        )
        while True:
            yield from loader


def get_train_dataset(dataset_name, T=None):
    return load_data(
        dataset_name, return_dataset=False, T=T,
        batch_size=None, deterministic=None, num_workers=None
    )


def get_test_dataset(dataset_name, T=None):
    if dataset_name == "mazes":
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if dataset_name == "minerl":
        data_path = os.path.join(data_path, "test")
        dataset = MineRLDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "test")
        dataset = GQNMazesDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "carla_no_traffic":
        dataset = CarlaDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


class BaseDataset(Dataset):
    """ The base class for our video datasets. It is used for datasets where each video is stored under <dataset_root_path>/<split>
        as a single file. This class provides the ability of caching the dataset items in a temporary directory (if
        specified as an environment variable DATA_ROOT) as the items are read. In other words, every time an item is
        retrieved from the dataset, it will try to load it from the temporary directory first. If it is not found, it
        will be first copied from the original location.

        This class provides a default implementation for __len__ as the number of file in the dataset's original directory.
        It also provides the following two helper functions:
        - cache_file: Given a path to a dataset file, makes sure the file is copied to the temporary directory. Does
        nothing unless DATA_ROOT is set.
        - get_video_subsequence: Takes a video and a video length as input. If the video length is smaller than the
          input video's length, it returns a random subsequence of the video. Otherwise, it returns the whole video.
        A child class should implement the following methods:
        - getitem_path: Given an index, returns the path to the video file.
        - loaditem: Given a path to a video file, loads and returns the video.
        - postprocess_video: Given a video, performs any postprocessing on the video.

    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path = self.getitem_path(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path)
        except Exception as e:
            print(f"Failed on loading {path}")
            raise e
        video = self.postprocess_video(video)
        return self.get_video_subsequence(video, self.T), {}

    def getitem_path(self, idx):
        raise NotImplementedError

    def loaditem(self, path):
        raise NotImplementedError

    def postprocess_video(self, video):
        raise NotImplementedError

    def cache_file(self, path):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            src_path = self.get_src_path(path)
            with Protect(path):
                shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """ Returns the source path to a file. This function is mainly used to handle SLURM_TMPDIR on ComputeCanada.
            If DATA_ROOT is defined as an environment variable, the datasets are copied to it as they are accessed. This function is called
            when we need the source path from a given path under DATA_ROOT.
        """
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            src_path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, T):
        if T is None:
            return video
        if T < len(video):
            # Take a subsequence of the video.
            start_i = 0 if self.is_test else np.random.randint(len(video) - T + 1)
            video = video[start_i:start_i+T]
        assert len(video) == T
        return video


class CarlaDataset(BaseDataset):
    def __init__(self, train, path, shard, num_shards, T):
        super().__init__(path=path, T=T)
        self.split_path = self.path / f"video_{'train' if train else 'test'}.csv"
        self.cache_file(self.split_path)
        self.fnames = [line.rstrip('\n').split('/')[-1] for line in open(self.split_path, 'r').readlines() if '.pt' in line]
        self.fnames = self.fnames[shard::num_shards]
        print(f"Loading {len(self.fnames)} files (Carla dataset).")

    def loaditem(self, path):
        return torch.load(path)

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    def postprocess_video(self, video):
        return -1 + 2 * (video.permute(0, 3, 1, 2).float()/255)

    def __len__(self):
        return len(self.fnames)


class GQNMazesDataset(BaseDataset):
    """ based on https://github.com/iShohei220/torch-gqn/blob/master/gqn_dataset.py .
    """
    def __init__(self, path, shard, num_shards, T):
        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f"{idx}.npy"

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        byte_to_tensor = lambda x: ToTensor()(x)
        video = torch.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video


class MineRLDataset(BaseDataset):
    def __init__(self, path, shard, num_shards, T):
        assert shard == 0, "Distributed training is not supported by the MineRL dataset yet."
        assert num_shards == 1, "Distributed training is not supported by the MineRL dataset yet."
        super().__init__(path=path, T=T)

    def getitem_path(self, idx):
        return self.path / f"{idx}.npy"

    def loaditem(self, path):
        return np.load(path)

    def postprocess_video(self, video):
        byte_to_tensor = lambda x: ToTensor()(x)
        video = torch.stack([byte_to_tensor(frame) for frame in video])
        video = 2 * video - 1
        return video
