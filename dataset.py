"""Loads dataset from Tensorflow datasets and local .mat files.

Normal instances are labelled 1 while anomalies are labelled 0.
"""

import os
from flax import nn
import jax
import jax.numpy as np
import scipy.io
import tensorflow_datasets as tfds

from tensorflow.io import gfile


def load_split(ds_name, split, pools=0, true_labels=[0], val=False):
  """Loads train, eval, or test split from Tensorflow image datasets.

  Images are reshaped to [N, H * W, C] and (2, 2) max-pooling is applied
  `pools` times. Sets labels falling into `true_labels` to `1` (normal) and
  all other labels to `0` (anomalous). If `val` is True and `split` =
  `tfds.Split.TRAIN`, 10% of the training set is used as the validation set.
  Only normal instances are retrieved for the train set.

  Args:
    ds_name: String denoting name of Tensorflow image dataset.
    split: tfds.Split instance, `tfds.Split.TRAIN` or `tfds.Split.TEST`.
    pools: Integer denoting number of max-pool operations.
    true_labels: List of Integers representing normal class labels.
    val: Bool for whether to extract a validation set from the train set.

  Returns:
    If `split` = `tfds.Split.TRAIN`, a duple of train and eval dictionaries
    with keys `data`, `label`, `original`.

    Else if `split` = `tfds.Split.TEST`, a test dictionary with keys `data`,
    `label`, `original`.

    `data`: Features with shape [N, H * W, C].
    `label`: `0` or `1` representing anomalous and normal.
    `original`: Integer corresponding to original classes.
  """
  ds = tfds.load(ds_name, split=split, batch_size=-1)
  data = tfds.as_numpy(ds)
  data['image'] = np.float32(data['image']) / 255.

  # Pool and reshape
  for _ in range(pools):
    data['image'] = nn.max_pool(data['image'], (2, 2),
                                strides=(2, 2), padding='SAME')
  d_shape = data['image'].shape
  data['image'] = np.reshape(data['image'],
                             (d_shape[0], d_shape[1] * d_shape[2], d_shape[3]))

  # Assign `1` to normal instances and `0` to anomalies
  data['original'] = data['label']
  labels = np.zeros(data['label'].shape, dtype=np.int32)
  for t in true_labels:
    labels += (data['label'] == t).astype(np.int32)
  data['label'] = labels

  if split == tfds.Split.TRAIN:
    # Only include normal instances in the training set.
    if val:
      # Extract 10% of the train set as the evaluation set.
      len_val = len(data['image']) // 10
      indices = data['label'][:-len_val] == 1
      train_data = {
          'data': data['image'][:-len_val][indices],
          'label': data['label'][:-len_val][indices],
          'original': data['original'][:-len_val][indices]
      }

      eval_data = {
          'data': data['image'][-len_val:],
          'label': data['label'][-len_val:],
          'original': data['original'][-len_val:]
      }
    else:
      indices = data['label'] == 1
      train_data = {
          'data': data['image'][indices],
          'label': data['label'][indices],
          'original': data['original'][indices]
      }
      eval_data = None
    return train_data, eval_data

  return data


def get_tfds_datasets(ds_name, pools=0, true_labels=[0], val=False):
  """Loads Tensorflow image datasets for anomaly detection.

  Training set consists of normal instances defined by `true_labels`. Evaluation
  dataset is None if `val` is False.

  Args:
    ds_name: String denoting name of Tensorflow image dataset.
    pools: Integer denoting number of max-pool operations.
    true_labels: List of Integers representing normal class labels.
    val: Bool for whether to extract a validation set from the train set.

  Returns:
    Tuple of three dictionaries corresponding to train, eval and test datasets.

    Dictionary keys are:
      `data`: Features with shape [N, H * W, C].
      `label`: `0` or `1` representing anomalous and normal.
      `original`: Integer corresponding to original classes.
  """
  train_ds, eval_ds = load_split(
      ds_name, tfds.Split.TRAIN, pools=pools, true_labels=true_labels, val=val)
  test_ds = load_split(ds_name, tfds.Split.TEST, pools=pools,
                       true_labels=true_labels)
  return train_ds, eval_ds, test_ds


def get_uci_datasets(ds_name, tab_dir, preprocess='normal'):
  """Loads UCI datasets from .mat files stored in directory `tab_dir`.

  Half of the normal instances is returned as the training set while the other
  half plus the anomalies is returned as the test set. Evaluation set is
  returned as None. Three types of preprocessing are available:

  `normal`: Normalizes features such that the training set has mean 0 and std 1.
  `interval_train`: Normalizes features such that the training set is in [0, 1].
  `interval_all`: Normalizes features such that all inputs are in [0, 1].

  Args:
    ds_name: String denoting name of UCI dataset, without `.mat` extension.
    tab_dir: String denoting directory where the dataset is stored.
    preprocess: String that is one of `normal`, `interval_train`, `interval_all`
                indicating type of preprocessing to perform.

  Returns:
    Tuple of dictionary, None, dictionary corresponding to train, eval and test
    datasets.

    Dictionary keys are:
      `data`: Features with shape [N, F, 1].
      `label`: `0` or `1` representing anomalous and normal.
  """
  base_path = '/cns/lu-d/home/jensenwang/data/'
  ds_path = base_path + ds_name + '.mat'
  ds_path = os.path.join(tab_dir, ds_name + '.mat')
  with gfile.GFile(ds_path, 'rb') as f:
    data = scipy.io.loadmat(f)
  samples = data['X']
  labels = ((data['y']).astype(np.int32)).reshape(-1)

  # UCI datasets represent normal instances as 0 by default, we negate that
  samples = samples[..., np.newaxis]
  labels = 1 - labels
  norm_samples = samples[labels == 1]
  anom_samples = samples[labels == 0]

  n_train = len(norm_samples) // 2
  x_train = norm_samples[:n_train]
  y_train = np.ones(x_train.shape[0], dtype=np.int32)
  test_real = norm_samples[n_train:]
  test_fake = anom_samples
  x_test = np.concatenate([test_real, test_fake], axis=0)
  y_test = np.concatenate([np.ones(test_real.shape[0], dtype=np.int32),
                           np.zeros(test_fake.shape[0], dtype=np.int32)],
                          axis=0)

  # Apply preprocessing
  if preprocess == 'normal':
    mu = np.mean(x_train, axis=0)
    sd = np.std(x_train, axis=0)
    sd = jax.ops.index_update(sd, sd == 0, 1)
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd
  elif 'interval' in preprocess:
    if preprocess == 'interval_train':
      x = x_train
    elif preprocess == 'interval_all':
      x = samples

    min_val = np.min(x, axis=0)
    max_val = np.max(x, axis=0)

    x_train = (x_train - min_val) / (max_val - min_val)
    x_test = (x_test - min_val) / (max_val - min_val)

  train_ds = {'data': x_train, 'label': y_train}
  test_ds = {'data': x_test, 'label': y_test}

  return train_ds, None, test_ds


def get_datasets(ds_name, pools=0, true_labels=[0], val=False,
                 tab_dir=None, preprocess='normal'):
  """Wrapper for loading Tensorflow and UCI datasets."""
  if ds_name in ['mnist', 'fashion_mnist', 'cifar10']:
    return get_tfds_datasets(
        ds_name, pools=pools, true_labels=true_labels, val=val)
  elif ds_name in ['thyroid', 'glass', 'wine', 'cover', 'satellite']:
    return get_uci_datasets(ds_name, tab_dir=tab_dir, preprocess=preprocess)
