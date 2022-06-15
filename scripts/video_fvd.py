from collections import defaultdict
from random import sample
from typing import OrderedDict
import torch
import numpy as np
import argparse
import os
from tqdm.auto import tqdm
from pathlib import Path
import json
import pickle
from collections import defaultdict
import tensorflow.compat.v1 as tf

# Metrics
from improved_diffusion.video_datasets import get_test_dataset
import improved_diffusion.frechet_video_distance as fvd
from improved_diffusion import test_util

tf.disable_eager_execution() # Required for our FVD computation code


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples_path, sample_idx, length):
        self.samples_path = Path(samples_path)
        self.sample_idx = sample_idx
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = self.samples_path / f"sample_{idx:04d}-{self.sample_idx}.npy"
        npy = np.load(path).astype(np.float32)
        normed = -1 + 2 * npy / 255
        return torch.tensor(normed).type(torch.float32), {}


class FVD:
    def __init__(self, batch_size, T, frame_shape):
        self.batch_size = batch_size
        self.vid = tf.placeholder("uint8", [self.batch_size, T, *frame_shape])
        self.vid_feature_vec = fvd.create_id3_embedding(fvd.preprocess(self.vid, (224, 224)), batch_size=self.batch_size)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.tables_initializer())

    def extract_features(self, vid):
        def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:
            # From here: https://stackoverflow.com/questions/19349410/how-to-pad-with-zeros-a-tensor-along-some-axis-python
            pad_size = target_length - array.shape[axis]
            if pad_size <= 0:
                return array
            npad = [(0, 0)] * array.ndim
            npad[axis] = (0, pad_size)
            return np.pad(array, pad_width=npad, mode='constant', constant_values=0)
        # vid is expected to have a shape of BxTxCxHxW
        B = vid.shape[0]
        vid = np.moveaxis(vid, 2, 4)  # B, T, H, W, C
        vid = pad_along_axis(vid, target_length=self.batch_size, axis=0)
        features = self.sess.run(self.vid_feature_vec, feed_dict={self.vid: vid})
        features = features[:B]
        return features

    @staticmethod
    def compute_fvd(vid1_features, vid2_features):
        return fvd.fid_features_to_metric(vid1_features, vid2_features)

def compute_fvd(test_dataset, sample_dataset, T, num_videos, batch_size=16):
    _, C, H, W = test_dataset[0][0].shape
    fvd_handler = FVD(batch_size=batch_size, T=T, frame_shape=[H, W, C])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    sample_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    assert len(test_dataset) == num_videos, f"{len(test_dataset)} != {num_videos}"
    assert len(sample_dataset) == num_videos, f"{len(sample_dataset)} != {num_videos}"
    with tf.Graph().as_default():
        all_test_features = []
        all_pred_features = []
        for (test_batch, _), (sample_batch, _) in zip(test_loader, sample_loader):
            scale = lambda x: ((x.numpy()+1)*255/2).astype(np.uint8)  # scale from [-1, 1] to [0, 255]
            test_batch = scale(test_batch)
            sample_batch = scale(sample_batch)
            test_features = fvd_handler.extract_features(test_batch)
            sample_features = fvd_handler.extract_features(sample_batch)
            all_test_features.append(test_features)
            all_pred_features.append(sample_features)
        all_test_features = np.concatenate(all_test_features, axis=0)
        all_pred_features = np.concatenate(all_pred_features, axis=0)
        fvd = fvd_handler.compute_fvd(all_pred_features, all_test_features)
    return fvd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--num_videos", type=int, default=None,
                        help="Number of generated samples per test video.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for extracting video features the I3D model.")
    parser.add_argument("--sample_idx", type=int, default=0)
    args = parser.parse_args()

    save_path = Path(args.eval_dir) / f"fvd-{args.num_videos}-{args.sample_idx}.txt"
    if save_path.exists():
        fvd = np.loadtxt(save_path).squeeze()
        print(f"FVD already computed: {fvd}")
        exit()

    # Load model args
    model_args_path = Path(args.eval_dir) / "model_config.json"
    with open(model_args_path, "r") as f:
        model_args = argparse.Namespace(**json.load(f))

    # Set batch size given dataset if not specified
    if args.batch_size is None:
        args.batch_size = {'mazes_cwvae': 16, 'minerl': 8, 'carla_no_traffic': 4}[model_args.dataset]

    # Prepare datasets
    sample_dataset = SampleDataset(samples_path=(Path(args.eval_dir) / "samples"), sample_idx=args.sample_idx, length=args.num_videos)
    test_dataset = torch.utils.data.Subset(
        dataset=get_test_dataset(dataset_name=model_args.dataset, T=model_args.T),
        indices=list(range(args.num_videos)),
    )
    fvd = compute_fvd(test_dataset, sample_dataset, T=model_args.T, num_videos=args.num_videos, batch_size=args.batch_size)
    np.savetxt(save_path, np.array([fvd]))
    print(f"FVD: {fvd}")