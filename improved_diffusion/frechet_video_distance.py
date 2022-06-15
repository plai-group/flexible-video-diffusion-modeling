# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal Reference implementation for the Frechet Video Distance (FVD).

FVD is a metric for the quality of video generation models. It is inspired by
the FID (Frechet Inception Distance) used for images, but uses a different
embedding to be better suitable for videos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import six
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import os
import numpy as np
import scipy

os.environ["TFHUB_CACHE_DIR"] = f"{os.environ['PWD']}/.tfhub_cache"


######################################################################
## Feature extraction                                               ##
######################################################################

def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution

  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def _is_in_graph(tensor_name):
  """Checks whether a given tensor does exists in the graph."""
  try:
    tf.get_default_graph().get_tensor_by_name(tensor_name)
  except KeyError:
    return False
  return True


def create_id3_embedding(videos, batch_size=16):
  """Embeds the given videos using the Inflated 3D Convolution network.

  Downloads the graph of the I3D from tf.hub and adds it to the graph on the
  first call.

  Args:
    videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
      Expected range is [-1, 1].

  Returns:
    embedding: <float32>[batch_size, embedding_size]. embedding_size depends
               on the model used.

  Raises:
    ValueError: when a provided embedding_layer is not supported.
  """

  module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


  # Making sure that we import the graph separately for
  # each different input video tensor.
  module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
      videos.name).replace(":", "_")

  assert_ops = [
      tf.Assert(
          tf.reduce_max(videos) <= 1.001,
          ["max value in frame is > 1", videos]),
      tf.Assert(
          tf.reduce_min(videos) >= -1.001,
          ["min value in frame is < -1", videos]),
      tf.assert_equal(
          tf.shape(videos)[0],
          batch_size, ["invalid frame batch size: ",
                       tf.shape(videos)],
          summarize=6),
  ]
  with tf.control_dependencies(assert_ops):
    videos = tf.identity(videos)

  module_scope = "%s_apply_default/" % module_name

  # To check whether the module has already been loaded into the graph, we look
  # for a given tensor name. If this tensor name exists, we assume the function
  # has been called before and the graph was imported. Otherwise we import it.
  # Note: in theory, the tensor could exist, but have wrong shapes.
  # This will happen if create_id3_embedding is called with a frames_placehoder
  # of wrong size/batch size, because even though that will throw a tf.Assert
  # on graph-execution time, it will insert the tensor (with wrong shape) into
  # the graph. This is why we need the following assert.
  video_batch_size = int(videos.shape[0])
  assert video_batch_size in [batch_size, -1, None], "Invalid batch size"
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  if not _is_in_graph(tensor_name):
    i3d_model = hub.Module(hub.resolve(module_spec), name=module_name)
    i3d_model(videos)

  # gets the kinetics-i3d-400-logits layer
  tensor_name = module_scope + "RGB/inception_i3d/Mean:0"
  tensor = tf.get_default_graph().get_tensor_by_name(tensor_name)
  return tensor


######################################################################
## Frechet distance computation                                     ##
######################################################################

# Adopted from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_fid.py
def frechet_statistics_from_features(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return {
        'mu': mu,
        'sigma': sigma,
    }

def frechet_statistics_to_frechet_metric(stat_1, stat_2):
    eps = 1e-6

    mu1, sigma1 = stat_1['mu'], stat_1['sigma']
    mu2, sigma2 = stat_2['mu'], stat_2['sigma']
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f'WARNING: fid calculation produces singular product; '
            f'adding {eps} to diagonal of cov estimates'
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=True)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, 'Imaginary component {}'.format(m)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    out = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

    return out

def fid_features_to_metric(features_1, features_2):
    assert isinstance(features_1, np.ndarray) and features_1.ndim == 2
    assert isinstance(features_2, np.ndarray) and features_2.ndim == 2
    assert features_1.shape[1] == features_2.shape[1]

    stat_1 = frechet_statistics_from_features(features_1)
    stat_2 = frechet_statistics_from_features(features_2)
    return frechet_statistics_to_frechet_metric(stat_1, stat_2)


######################################################################
## Kernel distance computation                                      ##
######################################################################

# Adopted from https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/metric_kid.py
KEY_METRIC_KID_MEAN = 'kernel_inception_distance_mean'
KEY_METRIC_KID_STD = 'kernel_inception_distance_std'

def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False, mmd_est='unbiased'):
    assert mmd_est in ('biased', 'unbiased', 'u-statistic'), 'Invalid value of mmd_est'

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
              + (Kt_YY_sum + sum_diag_Y) / (m * m)
              - 2 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2


def polynomial_kernel(X, Y, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = (np.matmul(X, Y.T) * gamma + coef0) ** degree
    return K


def polynomial_mmd(features_1, features_2, degree, gamma, coef0):
    k_11 = polynomial_kernel(features_1, features_1, degree=degree, gamma=gamma, coef0=coef0)
    k_22 = polynomial_kernel(features_2, features_2, degree=degree, gamma=gamma, coef0=coef0)
    k_12 = polynomial_kernel(features_1, features_2, degree=degree, gamma=gamma, coef0=coef0)
    return mmd2(k_11, k_12, k_22)


def kid_features_to_metric(features_1, features_2,
                           kid_subsets=100, kid_subset_size=1000,
                           kid_degree=3, kid_gamma=None, kid_coef0=1,
                           rng_seed=2020, verbose=False):
    # Default arguments from https://github.com/toshas/torch-fidelity/blob/a5cba01a8edc2b0f5303570e13c0f48eb6e96819/torch_fidelity/defaults.py
    assert isinstance(features_1, np.ndarray) and features_1.ndim == 2
    assert isinstance(features_2, np.ndarray) and features_2.ndim == 2
    assert features_1.shape[1] == features_2.shape[1]

    n_samples_1, n_samples_2 = len(features_1), len(features_2)
    assert \
        n_samples_1 >= kid_subset_size and n_samples_2 >= kid_subset_size,\
        f'KID subset size {kid_subset_size} cannot be smaller than the number of samples (input_1: {n_samples_1}, '\
        f'input_2: {n_samples_2}). Consider using "kid_subset_size" kwarg or "--kid-subset-size" command line key to '\
        f'proceed.'

    mmds = np.zeros(kid_subsets)
    rng = np.random.RandomState(rng_seed)

    for i in range(kid_subsets):
        f1 = features_1[rng.choice(n_samples_1, kid_subset_size, replace=False)]
        f2 = features_2[rng.choice(n_samples_2, kid_subset_size, replace=False)]
        o = polynomial_mmd(
            f1,
            f2,
            kid_degree,
            kid_gamma,
            kid_coef0,
        )
        mmds[i] = o

    out = {
        KEY_METRIC_KID_MEAN: float(np.mean(mmds)),
        KEY_METRIC_KID_STD: float(np.std(mmds)),
    }

    return out