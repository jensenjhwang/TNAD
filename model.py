"""TNAD model implementation."""

import functools

import jax
import flax
from flax import nn
import jax.numpy as np
import tensornetwork as tn

tn.set_default_backend('jax')
EPS = 1e-36

@jax.jit
def tensor_vectors(a, b):
  """Performs a tensor product operation along last dimension.

  Args:
    a: Tensor of shape [..., N].
    b: Tensor of shape [..., M].

  Returns:
    Tensor of shape [..., N*M] representing tensor product.
  """
  res = np.expand_dims(a, -1) * np.expand_dims(b, -2)
  new_shape = list(res.shape)[:-2] + [a.shape[-1] * b.shape[-1]]
  return np.reshape(res, new_shape)


def fourier_embedding(tensor, dim):
  """Performs `dim`-dimensional fourier embedding.

  Embedding causes `dim` number of uniformly separated values in [0,1] to be
  orthogonal. E.g. for dim = 4, values: 0, 1/3, 2/3, 1 will be orthogonal.
  Only works for a single channel.

  Args:
    tensor: Tensor of shape [N, F, 1].
    dim: Integer denoting dimension of embedding.

  Returns:
    Tensor of shape [N, F, `dim`].
  """
  comp = np.linspace(0, 1, dim, endpoint=False)[np.newaxis, np.newaxis, :]
  diff = (dim - 1) / dim * tensor - comp
  cur = np.zeros(diff.shape, dtype=np.complex64)
  for k in range(1, dim+1):
    cur += np.exp(2j * np.pi * k * diff)
  cur = np.abs(cur) / dim
  return cur


def trig_embedding(tensor, dim):
  """Performs `dim`-dimensional trigonometric embedding.

  `dim` must be even. If there are multiple channels in the input, the embedding
  is given by the tensor product of the embeddings over separate channels.

  Args:
    tensor: Tensor of shape [N, F, C].
    dim: Integer denoting dimension of embedding.

  Returns:
    Tensor of shape [N, F, `dim`**C].
  """
  emb_tensors = []

  for i in range(1, dim // 2 + 1):
    emb_tensors.append(np.sin(np.pi * tensor / (2 ** i)) / np.sqrt(dim // 2))
    emb_tensors.append(np.cos(np.pi * tensor / (2 ** i)) / np.sqrt(dim // 2))
  stacked_emb = np.stack(emb_tensors, -1)

  channel_embs = np.split(stacked_emb, stacked_emb.shape[-2], -2)
  channel_embs = [np.squeeze(t, axis=-2) for t in channel_embs]

  # Tensor product across channels
  cur = channel_embs[0]
  for t in channel_embs[1:]:
    cur = tensor_vectors(cur, t)

  return cur


def build_mps_or_mpo(tensors, periodic=False):
  """Creates MPS or MPO tensor network with appropriate boundary condition.

  Args:
    tensors: List of tensors of rank-3 for MPS and rank-4 for MPO.
    periodic: Bool indicating where to connect last node to the first.

  Returns:
    List of tn.Node representing connected network.
  """
  nodes = [tn.Node(t) for t in tensors]
  for i in range(len(nodes)-1):
    _ = nodes[i][-1] ^ nodes[i+1][0]
  if periodic:
    _ = nodes[-1][-1] ^ nodes[0][0]
  return nodes


def connect_node_list(nodes_1, nodes_2, axis=0, axis2=None):
  """Connects corresponding nodes in two lists along axes.

  The two lists must have the same length.

  Args:
    nodes_1: First list of tn.Node.
    nodes_2: Second list of tn.Node.
    axis: Integer denoting axis of `nodes_1` to be connected.
    axis2: Integer denoting axis of `nodes_2` to be connected, set to be `axis`
           if None.
  """
  assert len(nodes_1) == len(nodes_2), 'Node lists must have equal length.'
  if axis2 is None:
    axis2 = axis
  for n1, n2 in zip(nodes_1, nodes_2):
    _ = n1[axis] ^ n2[axis2]


def contract_node_list(nodes_1, nodes_2):
  """Contracts corresponding nodes in two lists.

  The two lists must have the same length.

  Args:
    nodes_1: First list of tn.Node.
    nodes_2: Second list of tn.Node.

  Returns:
    List of tn.Node representing contracted nodes.
  """
  assert len(nodes_1) == len(nodes_2), 'Node lists must have equal length.'
  new_list = []
  for n1, n2 in zip(nodes_1, nodes_2):
    new_list.append(n1 @ n2)
  return new_list


def contract_zig_zag(nodes_1, nodes_2, norm_interval):
  """Contracts two lists of tn.Node in a zig-zag fashion.

  Every `norm_interval` contractions, the node that is about to be contracted
  is divded by its max element. The sum of the logs of all rescaling operations
  is recorded and later returned.

  Args:
    nodes_1: First list of tn.Node.
    nodes_2: Second list of tn.Node.
    norm_interval: Integer denoting interval for normalizing tensors.

  Returns:
    Final node after contraction.
    Float denoting the sum of the logs of all rescaling operations.
  """
  assert len(nodes_1) == len(nodes_2), 'Node lists must have equal length.'
  cur_node = nodes_1[0]
  log_norm = 0
  for i in range(len(nodes_1)-1):
    if norm_interval is not None and i % norm_interval == 0:
      tensor = cur_node.get_tensor()
      scale = np.max(np.abs(tensor))
      log_norm += np.log(scale + EPS)
      cur_node.set_tensor(tensor / scale)
    cur_node = cur_node @ nodes_2[i]
    cur_node = cur_node @ nodes_1[i+1]
  cur_node = cur_node @ nodes_2[-1]
  return cur_node, log_norm


def connect_and_contract_zig_zag(nodes_1, nodes_2, skip_end_axis,
                                 norm_interval):
  """Connects two lists of tn.Node and contracts in a zig-zag fashion.

  If a node has rank R, all axes starting from index 1 to R-1-`skip_end_axis`
  are connected between the two node lists.

  Args:
    nodes_1: First list of tn.Node.
    nodes_2: Second list of tn.Node.
    skip_end_axis: Integer denoting number of trailing axes to ignore.
    norm_interval: Integer denoting interval for normalizing tensors.

  Returns:
    Final node after contraction.
    Float denoting the sum of the logs of all rescaling operations.
  """
  for i in range(1, nodes_1[0].get_rank() - 1 - skip_end_axis):
    connect_node_list(nodes_1, nodes_2, axis=i)
  node, log_norm = contract_zig_zag(nodes_1, nodes_2, norm_interval)
  return node, log_norm


def copy_list(nodes):
  """Returns a copy of a list of tn.Node."""
  nodes_dict, _ = tn.copy(nodes)
  new_nodes = [nodes_dict[node] for node in nodes]
  return new_nodes


def log_partition_fn(nodes, skip_end_axis=0, copy_original=False,
                     norm_interval=None):
  """Computes the log of the contraction of a list of nodes with themselves.

  Args:
    nodes: List of tn.node.
    skip_end_axis: Integer denoting number of trailing axes to ignore.
    copy_original: Bool indicating whether computation should be performed on
                   a duplicate of the original list to preserve it.
    norm_interval: Integer denoting interval for normalizing tensors.

  Returns:
    Float indicating log of the contracted network.
  """
  copy = copy_list(nodes)
  copy_2 = copy_list(nodes) if copy_original else nodes
  node, log_norm = connect_and_contract_zig_zag(copy_2, copy, skip_end_axis,
                                                norm_interval)
  log_Z = np.log(np.reshape(node.tensor, ()) + EPS) + log_norm
  return log_Z


@functools.partial(jax.jit, static_argnums=(1))
def log_partition_fn_mpo_mps(model_tensors, norm_interval):
  """Computes log of squared F-norm of a MPS or MPO.

  Args:
    model_tensors: List of tensors representing MPS (rank-3) or MPO (rank-4).
    norm_interval: Integer denoting interval for normalizing tensors.

  Returns:
    Float indicating log of squared F-norm.
  """
  model_nodes = build_mps_or_mpo(model_tensors)
  log_Z = log_partition_fn(model_nodes, norm_interval=norm_interval)
  return log_Z


@functools.partial(jax.jit, static_argnums=(2))
def mpo_prod(stacked_emb, model_tensors, norm_interval):
  """Computes log of the squared L2 norm upon transforming product state by MPO.

  Args:
    stacked_emb: Tensor of shape [N, F, P] representing product state.
    model_tensors: List of MPO tensors of rank-4.
     norm_interval: Integer denoting interval for normalizing tensors.

  Returns:
    Float representing logits.
  """
  emb_tensors = np.split(stacked_emb, stacked_emb.shape[0], 0)
  emb_nodes = [tn.Node(t) for t in emb_tensors]
  model_nodes = build_mps_or_mpo(model_tensors)

  connect_node_list(model_nodes, emb_nodes, axis=1)
  result_nodes = contract_node_list(model_nodes, emb_nodes)
  logits = log_partition_fn(result_nodes, skip_end_axis=1,
                            norm_interval=norm_interval)
  return logits


def contract_svd(model_tensors):
  """Computes the singular values of the MPO.

  Args:
    model_tensors: List of tensors of rank-4.

  Returns:
    List of floats representing singular values of MPO.
  """
  model_nodes = build_mps_or_mpo(model_tensors)
  copy_nodes = copy_list(model_nodes)
  connect_node_list(model_nodes, copy_nodes, 1)
  cur_node, log_norm = contract_zig_zag(model_nodes, copy_nodes, None)
  tensor = np.squeeze(cur_node.tensor)
  ndim = tensor.ndim
  cur_node = tn.Node(tensor)
  left_edges = [cur_node.get_edge(n) for n in range(0, ndim, 2)]
  right_edges = [cur_node.get_edge(n) for n in range(1, ndim, 2)]
  l, s, r, t = tn.split_node_full_svd(cur_node, left_edges, right_edges)
  singular_values = np.sqrt(np.diag(s.tensor) * np.exp(log_norm))
  return singular_values


class TNNet(nn.Module):
  """TNAD Model"""
  def apply(self, x, bond_dim=3, p_dim=2, spacing=8, m_dim=None,
            std=0.55, norm_interval=None, emb_type='trig'):

    emb_dim = p_dim ** (x.shape[-1])
    if m_dim is None:
      m_dim = emb_dim

    if emb_type == 'trig':
      assert(p_dim % 2 == 0), "Physical dimension must be even."
      stacked_emb = jax.jit(trig_embedding, static_argnums=(1,))(x, p_dim)
    elif emb_type == 'fourier':
      stacked_emb = jax.jit(fourier_embedding, static_argnums=(1,))(x, p_dim)

    sites = stacked_emb.shape[1]
    D = [1] + [bond_dim] * (sites - 1) + [1]

    model_tensors = [
        self.param(
            name='mpo_tensor{}'.format(n),
            shape=(D[n], emb_dim, m_dim, D[n + 1]),
            initializer=nn.initializers.normal(std)) if n %
        spacing == 0 else self.param(
            name='mpo_tensor{}'.format(n),
            shape=(D[n], emb_dim, 1, D[n + 1]),
            initializer=nn.initializers.normal(std)) for n in range(sites)
    ]

    log_Z = log_partition_fn_mpo_mps(model_tensors, norm_interval)

    logits = jax.jit(
        jax.vmap(mpo_prod, in_axes=(0, None, None), out_axes=0),
        static_argnums=(2,))(stacked_emb, model_tensors, norm_interval)

    return logits, log_Z


def create_model(key, params, sample_shape):
  module = TNNet.partial(bond_dim=params['bond_dim'],
                         p_dim=params['p_dim'],
                         spacing=params['spacing'],
                         m_dim=params['m_dim'],
                         std=params['std'],
                         norm_interval=params['norm_interval'],
                         emb_type=params['emb_type']
                         )
  """Creates flax.nn.Model instance of TNAD.

  Args:
    key: jax.random.PRNGKey for seeding.
    params: Dictionary of parameters.
    sample_shape: List of integers representing expected shape of inputs.

  Returns:
    flax.nn.Model instance of TNAD.
  """

  print('Received sample shape', sample_shape)
  _, initial_params = module.init_by_shape(key, [(sample_shape, np.float32)])

  model = nn.Model(module, initial_params)
  return model
