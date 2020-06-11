"""Training code for TNAD."""
import functools
import os
from datetime import datetime
import json

from absl import app
from absl import flags

import jax
import jax.numpy as np
import numpy as onpy
import sklearn.metrics
from flax import nn, optim
from flax.training import checkpoints

from jax.config import config

config.update('jax_disable_jit', True)

from tensorflow.io import gfile

from model import create_model
from dataset import get_datasets

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', default=2e-3,
    help=('The learning rate for the momentum optimizer.'))

flags.DEFINE_float(
    'momentum', default=0.9,
    help=('The decay rate used for the momentum optimizer.'))

flags.DEFINE_float(
    'alpha', default=0.3,
    help=('Weight for F-norm penalty.'))

flags.DEFINE_float(
    'std', default=0.5,
    help=('Standard deviation of normal initialization.'))

flags.DEFINE_integer(
    'batch_size', default=32,
    help=('Batch size for training.'))

flags.DEFINE_integer(
    'num_epochs', default=300,
    help=('Number of training epochs.'))

flags.DEFINE_string(
    'true_labels', default='0',
    help=('Non-anomalous label.'))

flags.DEFINE_string(
    'seeds', default='0,1,2,3,4,5,6,7,8,9',
    help=('Random seeds for network initialization.'))

flags.DEFINE_integer(
    'bond_dim', default=5,
    help=('Bond dimension of MPO.'))

flags.DEFINE_string(
    'emb_type', default='trig',
    help=('Embedding to use before TN model. `trig` or `fourier`.'))

flags.DEFINE_integer(
    'p_dim', default=2,
    help=('Physical dimension of MPO and embeddings.'))

flags.DEFINE_integer(
    'spacing', default=8,
    help=('Interval for MPO output, which determines dimension of output.'))

flags.DEFINE_integer(
    'm_dim', default=None,
    help=('Output dimension of MPO. If `m_dim` is None, it is set to `p_dim`.'))

flags.DEFINE_integer(
    'pool_image', default=1,
    help=('Number of times to half image dimensions by max-pooling.'))

flags.DEFINE_integer(
    'norm_interval', default=3,
    help=('Interval to re-normalize tensor during contraction.'))

flags.DEFINE_integer(
    'cold_start',
    default=20,
    help=('Number of initial epochs with reduced learning rate.'))

flags.DEFINE_float(
    'cold_factor', default=1e-2, help=('Learning rate reduction factor.'))

flags.DEFINE_string(
    'ds_name', default='mnist',
    help=('Datasets: `mnist`, `cifar10`.'))

flags.DEFINE_string(
    'tab_dir', default='./data/',
    help=('Directory where UCI tabular datasets are stored.'))

flags.DEFINE_string(
    'ckpt_dir', default='./save/',
    help=('Directory to store model data.'))

flags.DEFINE_string(
    'objective', default='roc',
    help=('Selection metric during evaluation.'))

flags.DEFINE_boolean(
    'val', default=False, help=('Whether to include evaluation set.'))

flags.DEFINE_float(
    'w_decay', default=0.01, help=('LR weight decay factor.'))

flags.DEFINE_string(
    'preprocess', default='normal',
    help=('Preprocessing step for tabular data.'))


def get_params(batch_size=32,
               num_epochs=30,
               learning_rate=5e-5,
               momentum=0.9,
               objective='roc',
               bond_dim=3,
               p_dim=2,
               pool_image=2,
               alpha=0.3,
               spacing=3,
               m_dim=None,
               seed=0,
               std=0.55,
               true_labels=[0],
               ds_name='mnist',
               norm_interval=None,
               cold_start=20,
               cold_factor=1e-2,
               val=False,
               w_decay=0.02,
               emb_type='trig',
               preprocess='normal',
               tab_dir='./data/'):
  """Returns dictionary of parameter keys and values."""
  return {
      'batch_size': batch_size,
      'num_epochs': num_epochs,
      'learning_rate': learning_rate,
      'momentum': momentum,
      'objective': objective,
      'bond_dim': bond_dim,
      'p_dim': p_dim,
      'pool_image': pool_image,
      'alpha': alpha,
      'spacing': spacing,
      'm_dim': m_dim,
      'seed': seed,
      'std': std,
      'true_labels': true_labels,
      'ds_name': ds_name,
      'norm_interval': norm_interval,
      'cold_start': cold_start,
      'cold_factor': cold_factor,
      'val': val,
      'w_decay': w_decay,
      'emb_type': emb_type,
      'preprocess': preprocess,
      'tab_dir': tab_dir
  }


def create_optimizer(model, learning_rate, beta):
  """Creates Adam optimizer for model.

  Args:
    model: flax.nn.Model to optimize.
    learning_rate: Adam optimizer learning rate.
    beta: Adam optimizer momentum.

  Returns:
    flax.optim.Momentum instance.
  """
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(model)
  return optimizer


@jax.jit
def train_step(optimizer, batch, alpha=1e-2):
  """Train step with squared error and F-norm penalty.

  Args:
    optimizer: flax.optim class instance.
    batch: Dictionary with keys `data` and `label` representing a single batch.
    alpha: Float denoting weight of F-norm penalty.

  Returns:
    flax.optim class instance representing updated optimizer.
    Integer representing the average batch loss.
  """
  def loss_fn(model):
    res, log_Z = model(batch['data'])
    reg = alpha * nn.relu(log_Z)
    se = np.mean((res - batch['label']) ** 2)
    loss = se + reg
    return loss

  loss, grad = jax.jit(jax.value_and_grad(loss_fn))(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss


def train_epoch(optimizer, train_ds, batch_size, rng, alpha=1e-2):
  """Train for a single epoch.

  Args:
    optimizer: flax.optim class instance.
    train_ds: Dictionary with keys `data` and `label`, representing train set.
    batch_size: Integer denoting batch size.
    rng: numpy.random.RandomState instance for seeding.
    alpha: Float denoting weight of F-norm penalty.

  Returns:
    flax.optim class instance representing updated optimizer.
  """
  train_ds_size = len(train_ds['data'])
  steps_per_epoch = train_ds_size // batch_size

  # Shuffle training set and batchify
  perms = rng.permutation(len(train_ds['data']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  for i, perm in enumerate(perms):
    batch = {k: v[perm] for k, v in train_ds.items()}
    optimizer, loss = train_step(optimizer, batch, alpha)
    if i % 20 == 0:
      print('Batch {} loss: {}'.format(i, loss))

  return optimizer


@jax.jit
def auc(fprs, tprs):
  """Computes area under curve.

   x-coordinates are `fprs` while y-coordinates are `tprs`.

  `fprs` and `tprs` must have the same length and `fprs` must be decreasing.

  Args:
    fprs: List of floats representing false positive rates, must be decreasing.
    tprs: List of floats representing true positive rates.

  Returns:
    Float in [0, 1] indicating area under curve.
  """
  x = np.array(fprs)
  y = np.array(tprs)
  x_step = x[:-1] - x[1:]
  y_avg = (y[1:] + y[:-1]) / 2
  res = np.sum(x_step * y_avg)
  return res


@jax.jit
def jittable_mean_slice(cond, array):
  """Jittable equivalent of np.mean(array[cond]).

  Args:
    cond: Bool mask.
    array: Tensor.

  Returns:
    Float representing mean of elements of `array` satisfying `cond`.
  """
  return np.sum(array * cond) / np.sum(cond)


@jax.jit
def compute_auc(logits, labels, thres_array=onpy.linspace(-100, 100, 200)):
  """Approximate AUC that is jittable.

  Use sklearn.metrics.roc_auc_score for exact scoring on test set.

  Args:
    logits: Float array denoting logits, with shape [N].
    labels: Array of `0` or `1` indicating labels, with shape [N].
    thres_array: Float array of thresholds determining points on the roc.

  Returns:
    Float list of false positive rates at the given thresholds.
    Float list of true positive rates at the given thresholds.
    Float in [0, 1] indicating area under ROC.
  """
  # Endpoint of ROC
  fprs = [1.]
  tprs = [1.]
  for thres in thres_array:
    pred = (logits > thres).astype(np.int32)
    equal = (pred == labels).astype(np.int32)
    pos_recall = jittable_mean_slice(labels == 1, equal)
    neg_recall = jittable_mean_slice(labels == 0, equal)
    fprs.append(1 - neg_recall)
    tprs.append(pos_recall)

  # Other endpoint of ROC
  fprs.append(0.)
  tprs.append(0.)

  roc_auc = auc(fprs, tprs)

  return fprs, tprs, roc_auc


@jax.jit
def compute_metrics(logits, labels):
  """Computes metrics from logits and labels.

  Args:
    logits: Float array denoting logits, with shape [N].
    labels: Array of `0` or `1` indicating labels, with shape [N].

  Returns:
    Dictionary of metrics.
  """
  pred = (logits > 0.5).astype(np.int32)
  loss = np.mean((logits - labels)**2)  # without F-norm penalty

  equal = (pred == labels).astype(np.int32)
  result = np.stack([labels, pred, equal], axis=-1)

  acc = np.mean(result[:, 2])

  pos_recall = jittable_mean_slice(labels == 1, equal)
  neg_recall = jittable_mean_slice(labels == 0, equal)
  pos_precision = jittable_mean_slice(pred == 1, equal)
  neg_precision = jittable_mean_slice(pred == 0, equal)

  _, _, auc_roc = compute_auc(logits, labels)

  metrics = {
      'loss': loss,
      'accuracy': acc,
      'pos_precision': pos_precision,
      'pos_recall': pos_recall,
      'pos_f1': 2 * pos_precision*pos_recall / (pos_precision+pos_recall),
      'neg_precision': neg_precision,
      'neg_recall': neg_recall,
      'neg_f1': 2 * neg_precision*neg_recall / (neg_precision+neg_recall),
      'roc': auc_roc
  }

  return metrics


def eval_model(model, ds, test=False):
  """Returns evaluation metrics on dataset.

  If `test` is True, adds additional metrics `roc2` which is a more precise
  computation of the AUROC given by sklearn.metrics.roc_auc_score, `prcin`
  and `prcout` which are the areas under the PRCIn and PRCOut curves.

  Args:
    model: flax.nn.Model instance.
    ds: Dictionary with keys `data` and `label` to evaluate on.
    test: Bool indicating whether dataset is the test set.

  Returns:
    Dictionary of metrics.
  """
  logits, log_Z = model(ds['data'])
  metrics = compute_metrics(logits, ds['label'])
  if test:
    # Skip if there are invalid entries in logits
    if np.sum(np.isnan(logits) + np.isinf(logits)) == 0:
      metrics['roc2'] = sklearn.metrics.roc_auc_score(ds['label'], logits)
      p, r, t = sklearn.metrics.precision_recall_curve(ds['label'], logits)
      metrics['prcin'] = sklearn.metrics.auc(r, p)
      p, r, t = sklearn.metrics.precision_recall_curve(
          ds['label'], -logits, pos_label=0)
      metrics['prcout'] = sklearn.metrics.auc(r, p)
    else:
      metrics['roc2'] = None
      metrics['prcin'] = None
      metrics['prcout'] = None
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary


def test_train(optimizer, train_ds, batch_size, alpha):
  """Trains model for one batch and returns if training failed.

  Training fails if loss if nan or inf.

  Args:
    optimizer: flax.optim class instance.
    train_ds: Dictionary with keys `data` and `label`, representing train set.
    batch_size: Integer denoting batch size.
    alpha: Float denoting weight of F-norm penalty.

  Returns:
    Bool indicating if training failed.
  """
  batch = {k: v[:batch_size] for k, v in train_ds.items()}
  optimizer, loss = train_step(optimizer, batch, alpha)
  return np.isnan(loss) or np.isinf(loss)


def train_and_evaluate(train_ds, eval_ds, test_ds, ckpt_dir, params, seed=0):
  """Train model to completion, evaluating every epoch.

  If eval_ds is None, the model is not evaluated. Otherwise, the model is
  evaluted every epoch on eval_ds and the model with the highest value of
  `params['objective']` is tested on the test set.

  Args:
    train_ds: Dictionary with keys `data` and `label`, representing train set.
    eval_ds: Dictionary with keys `data` and `label`, representing eval set.
    test_ds: Dictionary with keys `data` and `label`, representing test set.
    ckpt_dir: String indicating directory where checkpoints should be stored.
    params: Dictionary of parameters.
    seed: Integer for random seeds.

  Returns:
    Dictionary of metrics on the test set.
  """
  rng = jax.random.PRNGKey(seed)

  sample_shape = [1] + list(train_ds['data'][0].shape)

  # Attempt to train for a batch and decrease initialization std if it fails.
  tries = 10
  is_nan = True
  for _ in range(tries):
    model = create_model(rng, params, sample_shape)
    optimizer = create_optimizer(model, params['learning_rate'],
                                 params['momentum'])
    if not test_train(optimizer, train_ds, params['batch_size'],
                      params['alpha']):
      is_nan = False
      break
    print('Test train failed, decreasing std.')
    params['std'] -= 0.025

  if is_nan:
    print('Failed to train')
    return {params['objective']: 0}

  # Reload model if it already exists; save parameters otherwise
  if gfile.exists(os.path.join(ckpt_dir, 'params.json')):
    print('Found existing model, reloading params...')
    with gfile.GFile(os.path.join(ckpt_dir, 'params.json')) as f:
      params = json.load(f)
    optimizer = checkpoints.restore_checkpoint(ckpt_dir, optimizer)
  else:
    if not gfile.exists(ckpt_dir):
      gfile.makedirs(ckpt_dir)
    with gfile.GFile(os.path.join(ckpt_dir, 'params.json'), 'w') as f:
      json.dump(params, f)

  objective = params['objective']

  print('Initial evaluation')
  train_metrics = eval_model(optimizer.target, train_ds)
  print('Train', train_metrics)
  if eval_ds is not None:
    best_eval_metrics = eval_model(optimizer.target, eval_ds)
    best_obj = 0
    print('Eval', best_eval_metrics)

  input_rng = onpy.random.RandomState(seed)
  lr = params['learning_rate']
  cold_factor = params['cold_factor']

  for epoch in range(1, params['num_epochs'] + 1):
    print('Epoch {}'.format(epoch))
    # Cold start settings
    if params['cold_start'] > 0:
      if epoch == 1:
        optimizer.optimizer_def.update_hyper_params(learning_rate=lr *
                                                    cold_factor)
      elif epoch > params['cold_start']:
        exp_factor = np.exp(-params['w_decay'] *
                            (epoch - params['cold_start'] - 1))
        optimizer.optimizer_def.update_hyper_params(learning_rate=lr *
                                                    exp_factor)
    # Train for an epoch
    optimizer = train_epoch(optimizer, train_ds, params['batch_size'],
                            input_rng, alpha=params['alpha'])

    # Evaluation
    print('Evaluating...')
    train_metrics = eval_model(optimizer.target, train_ds)
    print('Train', train_metrics)

    if eval_ds is not None:
      metrics = eval_model(optimizer.target, eval_ds)
      print('Eval', metrics)
      obj = metrics[objective]

      if obj > best_obj:
        # Save best model found so far
        best_obj = obj
        checkpoints.save_checkpoint(ckpt_dir, optimizer, epoch, keep=2)
        print('New best found, saving model...')
        best_eval_metrics = metrics

  # Test
  print('Evaluating on test set.')
  if eval_ds is not None:
    # Reload best model found during evaluation, if eval_ds is not None.
    optimizer = checkpoints.restore_checkpoint(ckpt_dir, optimizer)
    with gfile.GFile(os.path.join(ckpt_dir, 'results.json'), 'w') as f:
      json.dump(best_eval_metrics, f)
  else:
    # Else eval_ds is None, save model after final training epoch.
    checkpoints.save_checkpoint(
        ckpt_dir, optimizer, params['num_epochs'], keep=2)

  # Test metrics
  test_metrics = eval_model(optimizer.target, test_ds, test=True)
  print(test_metrics)
  with gfile.GFile(os.path.join(ckpt_dir, 'test_results.json'), 'w') as f:
    json.dump(test_metrics, f)

  print(params)
  print(ckpt_dir)

  return test_metrics


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  now = datetime.now()
  dt_string = now.strftime('%d_%m_%Y_%H_%M_%S')

  averages = {'roc2': [], 'prcin': [], 'prcout': []}
  success_seeds = []
  true_labels = [int(s) for s in FLAGS.true_labels.split(',')]
  seeds = [int(s) for s in FLAGS.seeds.split(',')]
  for seed in seeds:
    ckpt_dir = FLAGS.ckpt_dir + '{}_{}/'.format(dt_string, seed)
    params = get_params(
        batch_size=FLAGS.batch_size,
        num_epochs=FLAGS.num_epochs,
        learning_rate=FLAGS.learning_rate,
        momentum=FLAGS.momentum,
        objective=FLAGS.objective,
        bond_dim=FLAGS.bond_dim,
        p_dim=FLAGS.p_dim,
        pool_image=FLAGS.pool_image,
        alpha=FLAGS.alpha,
        spacing=FLAGS.spacing,
        m_dim=FLAGS.m_dim,
        seed=seed,
        true_labels=true_labels,
        ds_name=FLAGS.ds_name,
        std=FLAGS.std,
        norm_interval=FLAGS.norm_interval,
        cold_start=FLAGS.cold_start,
        cold_factor=FLAGS.cold_factor,
        val=FLAGS.val,
        w_decay=FLAGS.w_decay,
        emb_type=FLAGS.emb_type,
        preprocess=FLAGS.preprocess,
        tab_dir=FLAGS.tab_dir)

    train_ds, eval_ds, test_ds = get_datasets(
        params['ds_name'],
        pools=params['pool_image'],
        true_labels=params['true_labels'],
        val=params['val'],
        preprocess=params['preprocess'],
        tab_dir=params['tab_dir'])

    test_metrics = train_and_evaluate(
        train_ds,
        eval_ds,
        test_ds,
        ckpt_dir,
        params=params,
        seed=params['seed'])

    if test_metrics['roc2'] is not None:
      success_seeds.append(seed)
      for key in averages:
        averages[key].append(test_metrics[key])

  print('Finished training')
  for key in averages:
    mean = np.mean(np.asarray(averages[key][:10]))
    std = np.std(np.asarray(averages[key][:10]))
    print(averages[key])
    print('{}: '.format(key), mean, '+/-', std)

  print('Success seeds: ', success_seeds)


if __name__ == '__main__':
  app.run(main)
