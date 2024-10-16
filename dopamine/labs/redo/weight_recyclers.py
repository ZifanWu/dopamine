# coding=utf-8
# Copyright 2023 ReDo authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This class implements recycling of dead neurons."""
import functools
import logging
import flax
from flax import linen as nn
import gin
import jax
from jax import random
import jax.numpy as jnp
import optax
import wandb
# import jaxpruner

from dopamine.discrete_domains.run_experiment import config


def leastk_mask(scores, ones_fraction):
  """Given a tensor of scores creates a binary mask.

  Args:
    scores: top-scores are kept
    ones_fraction: float, of the generated mask.

  Returns:
    array, same shape and type as scores or None.
  """
  if ones_fraction is None or ones_fraction == 0:
    return jnp.zeros_like(scores)
  # This is to ensure indices with smallest values are selected.
  scores = -scores

  n_ones = jnp.round(jnp.size(scores) * ones_fraction)
  k = jnp.maximum(1, n_ones).astype(int)
  flat_scores = jnp.reshape(scores, -1)
  threshold = jax.lax.sort(flat_scores)[-k]

  mask = (flat_scores >= threshold).astype(flat_scores.dtype)
  return mask.reshape(scores.shape)


def reset_momentum(momentum, mask):
  new_momentum = momentum if mask is None else momentum * (1.0 - mask)
  return new_momentum


def weight_reinit_zero(param, mask):
  if mask is None:
    return param
  else:
    new_param = jnp.zeros_like(param)
    param = jnp.where(mask == 1, new_param, param)
    return param


def weight_reinit_random(
    param, mask, key, weight_scaling=False, scale=1.0, weights_type='incoming'
):
  """Randomly reinit recycled weights and may scale its norm.

  If scaling applied, the norm of recycled weights equals
  the average norm of non recycled weights per neuron multiplied by a scalar.

  Args:
    param: current param
    mask: incoming/outgoing mask for recycled weights
    key: random key to generate new random weights
    weight_scaling: if true scale recycled weights with the norm of non recycled
    scale: scale to multiply the new weights norm.
    weights_type: incoming or outgoing weights

  Returns:
  params: new params after weight recycle.
  """
  if mask is None or key is None:
    return param

  new_param = nn.initializers.xavier_uniform()(key, shape=param.shape)

  if weight_scaling:
    axes = list(range(param.ndim))
    if weights_type == 'outgoing':
      del axes[-2]
    else:
      del axes[-1]

    neuron_mask = jnp.mean(mask, axis=axes)

    non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    non_recycled_norm = (
        jnp.sum(norm_per_neuron * (1 - neuron_mask)) / non_dead_count
    )
    non_recycled_norm = non_recycled_norm * scale

    normalized_new_param = _weight_normalization_per_neuron_norm(
        new_param, axes
    )
    new_param = normalized_new_param * non_recycled_norm

  param = jnp.where(mask == 1, new_param, param)
  return param


def _weight_normalization_per_neuron_norm(param, axes):
  norm_per_neuron = _get_norm_per_neuron(param, axes)
  norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
  normalized_param = param / norm_per_neuron
  return normalized_param


def _get_norm_per_neuron(param, axes):
  return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))


@gin.configurable
class BaseRecycler:
  """Base class for weight update methods.

  Attributes:
    all_layers_names: list of layer names in a model.
    recycle_type: neuron, layer based.
    dead_neurons_threshold: below this threshold a neuron is considered dead.
    reset_layers: list of layer names to be recycled.
    reset_start_layer_idx: index of the layer from which we start recycling.
    reset_period: int represents the period of weight update.
    reset_start_step: start recycle from start step
    reset_end_step:  end recycle from end step
    logging_period:  the period of statistics logging e.g., dead neurons.
    prev_neuron_score: score at last reset step or log step in case of no reset.
    sub_mean_score: if True the average activation will be subtracted for each
      neuron when we calculate the score.
  """

  def __init__(
      self,
      all_layers_names,
      dead_neurons_threshold=0.1,
      reset_start_layer_idx=0,
      reset_period=200_000,
      reset_start_step=0,
      reset_end_step=100_000_000,
      logging_period=20_000,
      sub_mean_score=False,
  ):
    self.all_layers_names = all_layers_names
    self.dead_neurons_threshold = dead_neurons_threshold
    self.reset_layers = all_layers_names[reset_start_layer_idx:]
    self.reset_period = reset_period
    self.reset_start_step = reset_start_step
    self.reset_end_step = reset_end_step
    self.logging_period = logging_period
    self.prev_neuron_score = None
    self.sub_mean_score = sub_mean_score

    # NOTE (ZW) added
    self.historical_dormant_mask = None

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]

  def is_update_iter(self, step):
    return step > 0 and (step % self.reset_period == 0)

  def update_weights(self, intermediates, params, key, opt_state):
    raise NotImplementedError

  def maybe_update_weights(
      self, update_step, intermediates, params, key, opt_state
  ):
    self._last_update_step = update_step
    if self.is_reset(update_step):
      new_params, new_opt_state = self.update_weights(
          intermediates, params, key, opt_state
      )
    else:
      new_params, new_opt_state = params, opt_state
    return new_params, new_opt_state

  def is_reset(self, update_step):
    del update_step
    return False

  def is_intermediated_required(self, update_step):
    return self.is_logging_step(update_step)

  def is_logging_step(self, step):
    return step % self.logging_period == 0

  def maybe_log_deadneurons(self, update_step, intermediates):
    is_logging = self.is_logging_step(update_step)
    if is_logging:
      return self.log_dead_neurons_count(intermediates, update_step)
    else:
      return None

  def intersected_dead_neurons_with_last_reset(
      self, intermediates, update_step
  ):
    if self.is_logging_step(update_step):
      log_dict = self.log_intersected_dead_neurons(intermediates, update_step)
      self.log_historical_dead_neuron_overlapping(intermediates, update_step)
      return log_dict
    else:
      return None

  def log_intersected_dead_neurons(self, intermediates, update_step):
    """Track intersected dead neurons with last logging/reset step.

    Args:
      intermediates: current intermediates

    Returns:
      log_dict: dict contains the percentage of intersection
    """
    score_tree = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep='/')

    if self.prev_neuron_score is None:
      self.prev_neuron_score = neuron_score_dict
      log_dict = None
    else:
      log_dict = {}
      for prev_k_score, current_k_score in zip(
          self.prev_neuron_score.items(), neuron_score_dict.items()
      ):
        _, prev_score = prev_k_score
        k, score = current_k_score
        prev_score, score = prev_score[0], score[0]
        prev_mask = prev_score <= self.dead_neurons_threshold
        # we count the dead neurons which remains dead in the current step.
        intersected_mask = (prev_mask) & (score <= self.dead_neurons_threshold)
        prev_dead_count = jnp.count_nonzero(prev_mask)
        intersected_count = jnp.count_nonzero(intersected_mask)

        percent = (
            (float(intersected_count) / prev_dead_count)
            if prev_dead_count
            else 0.0
        )
        log_dict[f'dead_intersected_percent/{k[:-9]}'] = float(percent) * 100.0

        # track average score of recycled neurons from last step and
        # the average of non dead in the current step.
        nondead_mask = score > self.dead_neurons_threshold
        log_dict[f'mean_score_recycled/{k[:-9]}'] = float(
            jnp.mean(score[prev_mask])
        )
        log_dict[f'mean_score_nondead/{k[:-9]}'] = float(
            jnp.mean(score[nondead_mask])
        )
        if config['use_wandb']:
          wandb.log({'{}_dead_intersected_percent'.format(k[:-9]): percent, 'grad_step': update_step})

      self.prev_neuron_score = neuron_score_dict
    return log_dict
  
  def log_historical_dead_neuron_overlapping(self, intermediates, update_step):
    """Track the overlapping rate of dead neurons between the historical set/and the current step.

    Args:
      intermediates: current intermediates

    Returns:
      log_dict: dict contains the percentage of intersection
    """
    score_tree = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    # for k in score_tree.keys():
    #   print(11, k)
    #   for k1 in score_tree[k].keys():
    #     print(22, len(score_tree[k][k1]))
    #     print(score_tree[k][k1][0].shape)
    # 11 Conv_0_act
    # 22 1
    # (32,) # in Conv layers, the number of neurons is the size of the feature map
    # 11 Conv_1_act
    # 22 1
    # (64,)
    # 11 Conv_2_act
    # 22 1
    # (64,)
    # 11 Dense_0_act
    # 22 1
    # (512,)
    # 11 final_layer_act
    # 22 1
    # (6,)
    neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep='/')
    # for k in neuron_score_dict.keys():
    #   print(11, k, len(neuron_score_dict[k]))
    #   for i in range(len(neuron_score_dict[k])):
    #     print(22, neuron_score_dict[k][i].shape)
    # 11 Conv_0_act/__call__ 1
    # 22 (32,)
    # 11 Conv_1_act/__call__ 1
    # 22 (64,)
    # 11 Conv_2_act/__call__ 1
    # 22 (64,)
    # 11 Dense_0_act/__call__ 1
    # 22 (512,)
    # 11 final_layer_act/__call__ 1
    # 22 (6,)

    if self.historical_dormant_mask is None:
      self.prev_neuron_score = neuron_score_dict
      log_dict = None
      self.historical_dormant_mask = {} # recording neurons that have at least once been detected dormant (whose entries take True)
      self.dormant_times = {} # recording the times each neuron is detected dormant
      # self.degree_of_dormancy = {} # recording (dormant_times) / (logging_times)
      self.n_log_historical_overlap = 1
    else:
      self.n_log_historical_overlap += 1
      log_dict = {}
      for prev_k_score, current_k_score in zip(
          self.prev_neuron_score.items(), neuron_score_dict.items()
      ): # layer k
        # print(prev_k_score[0], prev_k_score[1][0].shape) # Conv_0_act/__call__ (32,)
        _, prev_score = prev_k_score
        k, score = current_k_score
        prev_score, score = prev_score[0], score[0]
        prev_mask = prev_score <= self.dead_neurons_threshold
        # we count the dead neurons which remains dead in the current step.
        curr_mask = score <= self.dead_neurons_threshold

        if k not in self.historical_dormant_mask.keys(): # first log
          self.historical_dormant_mask[k] = prev_mask # non-dormant entries: False
          self.dormant_times[k] = jnp.zeros_like(prev_mask).astype(float)
          # self.degree_of_dormancy[k] = jnp.zeros_like(curr_mask).astype(float)
        self.dormant_times[k] += curr_mask.astype(float)
        # TODO (ZW) what if we reset a neuron only if its degree_of_dormancy has reached a threshold
        degree_of_dormancy = self.dormant_times[k] / self.n_log_historical_overlap
        # avg_degree_of_dormancy = degree_of_dormancy.mean()

        pre_hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        self.historical_dormant_mask[k] = (self.historical_dormant_mask[k]) | (curr_mask) # NOTE (ZW) merging the current dormant set into the historical set

        intersected_mask = (self.historical_dormant_mask[k]) & (curr_mask)
        intersected_count = jnp.count_nonzero(intersected_mask)
        curr_dead_count = jnp.count_nonzero(curr_mask)
        # hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        post_hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        denominator = post_hist_dead_count # This implements the post-merging-set-as-denominator metric
        # denominator = max(curr_dead_count, hist_dead_count) # This implements the max-pre-merging-set-as-denominator metric

        # self.historical_dormant_mask[k] = (self.historical_dormant_mask[k]) | (curr_mask)

        percent = (
            (float(intersected_count) / denominator.item())
            if denominator
            else 0.0
        )
        print('percent: {}'.format(percent))
        log_dict[f'historical_overlap_rate/{k[:-9]}'] = float(percent) * 100.0

        # track average score of recycled neurons from last step and
        # the average of non dead in the current step.
        # nondead_mask = score > self.dead_neurons_threshold
        # log_dict[f'mean_score_recycled/{k[:-9]}'] = float(
        #     jnp.mean(score[prev_mask])
        # )
        # log_dict[f'mean_score_nondead/{k[:-9]}'] = float(
        #     jnp.mean(score[nondead_mask])
        # )
        if config['use_wandb']:
          wandb.log({'{}_historical_overlap_rate'.format(k[:-9]): percent, 'grad_step': update_step})
          wandb.log({'{}_current_historical_ratio(pre_merging)'.format(k[:-9]): (curr_dead_count / pre_hist_dead_count).item(), 'grad_step': update_step})
          wandb.log({'{}_historical_dormant_count(post_merging)'.format(k[:-9]): post_hist_dead_count.item(), 'grad_step': update_step})

      self.prev_neuron_score = neuron_score_dict
    return log_dict

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_neurons_threshold

  def log_outgoing_weights_magnitude(self, param_dict, activations_dict, key):
    """log: # TODO (ZW)
    1. magnitude of dormant neurons's outgoing_weights (in each layer)
    2. magnitude of the rest neurons' outgoing_weights in the same layer
    """
    incoming_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    outgoing_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    ingoing_random_keys_dict = {k: None for k in param_dict}
    outgoing_random_keys_dict = (
        {k: None for k in param_dict}
        if self.init_method_outgoing == 'random'
        else {}
    )

    # prepare mask of incoming and outgoing recycled connections
    for k in self.reset_layers:
      param_key = 'params/' + k + '/kernel'
      param = param_dict[param_key]
      # This won't work for DRQ, since returned keys can be a list.
      # We don't support that at the moment.
      next_key = self.next_layers[k]
      if isinstance(next_key, list):
        next_key = next_key[0]
      next_param = param_dict['params/' + next_key + '/kernel']
      activation = activations_dict[k + '_act/__call__'][0]
      # TODO(evcu) Maybe use per_layer random keys here.
      neuron_mask = self._score2mask(activation, param, next_param, key)

      # the for loop handles the case where a layer has multiple next layers
      # like the case in DrQ where the output layer has multihead.
      next_keys = (
          self.next_layers[k]
          if isinstance(self.next_layers[k], list)
          else [self.next_layers[k]]
      )
      for next_k in next_keys:
        next_param_key = 'params/' + next_k + '/kernel'
        next_param = param_dict[next_param_key]
        incoming_mask, outgoing_mask = self.create_mask_helper(
            neuron_mask, param, next_param
        )
        incoming_mask_dict[param_key] = incoming_mask
        outgoing_mask_dict[next_param_key] = outgoing_mask
        key, subkey = random.split(key)
        ingoing_random_keys_dict[param_key] = subkey
        if self.init_method_outgoing == 'random':
          key, subkey = random.split(key)
          outgoing_random_keys_dict[next_param_key] = subkey

    #     if self.prune_dormant_neurons: # NOTE (ZW) stop the gradients flowing through dormant neurons
    #       # NOTE (ZW) Log the magnitude of outgoing weights of dormant neurons
    #       print('Pruning {} outgoing weights at layer {}'.format(outgoing_mask.sum(), k))
    #       next_param = jnp.where(~outgoing_mask, next_param, jax.lax.stop_gradient(next_param))
        
    #       if self.first_time_pruning and (jnp.count_nonzero(outgoing_mask) > 0):
    #         self.next_param_key = next_param_key
    #         self.outgoing_mask = outgoing_mask
    #         self.first_time_pruning = False

    # if (not self.first_time_pruning) and self.prune_dormant_neurons:
    #   print(jnp.count_nonzero(self.outgoing_mask), self.next_param_key)
    #   frozen_params = jnp.where(self.outgoing_mask == 1, jnp.zeros_like(param_dict[self.next_param_key]), param_dict[self.next_param_key])
    #   print(jnp.linalg.vector_norm(frozen_params).item())
    #   import time
    #   time.sleep(2)
    #   if config['use_wandb']:
    #     wandb.log({'frozen_params_norm': jnp.linalg.vector_norm(frozen_params).item(), 'grad_step': self._last_update_step})

      # reset bias
      bias_key = 'params/' + k + '/bias'
      new_bias = jnp.zeros_like(param_dict[bias_key])
      param_dict[bias_key] = jnp.where(
          neuron_mask, new_bias, param_dict[bias_key]
      ) # True entities in param_dict[bias_key] will be replaced by new_bias

    return (
        incoming_mask_dict,
        outgoing_mask_dict,
        ingoing_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    )


  def log_dead_neurons_count(self, intermediates, update_step):
    """log dead neurons in each layer.

    For conv layer we also log dead elements in the spatial dimension.

    Args:
      intermediates: intermidate activation in each layer.

    Returns:
      log_dict_elements_per_neuron
      log_dict_neurons
    """

    def log_dict(score, score_type):
      total_neurons, total_deadneurons = 0.0, 0.0
      score_dict = flax.traverse_util.flatten_dict(score, sep='/')

      log_dict = {}
      layer_count = 0
      for k, m in score_dict.items():
        layer_count += 1
        if 'final_layer' in k:
          continue
        m = m[0]
        layer_size = float(jnp.size(m))
        deadneurons_count = jnp.count_nonzero(m <= self.dead_neurons_threshold)
        total_neurons += layer_size
        total_deadneurons += deadneurons_count
        log_dict[f'dead_{score_type}_percentage/{k[:-9]}'] = (
            float(deadneurons_count) / layer_size
        ) * 100.0
        log_dict[f'dead_{score_type}_count/{k[:-9]}'] = float(deadneurons_count)
        if config['use_wandb']:
          wandb.log({'layer {} dormant neuron percentage'.format(layer_count): float(deadneurons_count) / layer_size, 'grad_step': update_step})
      log_dict[f'{score_type}/total'] = total_neurons
      log_dict[f'{score_type}/deadcount'] = float(total_deadneurons)
      log_dict[f'dead_{score_type}_percentage'] = (
          float(total_deadneurons) / total_neurons
      ) * 100.0
      if config['use_wandb']:
        wandb.log({'overall dormant neuron percentage': float(total_deadneurons) / total_neurons, 'grad_step': update_step})
      return log_dict

    neuron_score = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    log_dict_neurons = log_dict(neuron_score, 'feature')

    return log_dict_neurons

  def estimate_neuron_score(self, activation, is_cbp=False):
    """Calculates neuron score based on absolute value of activation.

    The score of feature map is the normalized average score over
    the spatial dimension.

    Args:
      activation: intermediate activation of each layer
      is_cbp: if true, subtracts the mean and skips normalization.

    Returns:
      element_score: score of each element in feature map in the spatial dim.
      neuron_score: score of feature map
    """
    reduce_axes = list(range(activation.ndim - 1))
    if self.sub_mean_score or is_cbp:
      activation = activation - jnp.mean(activation, axis=reduce_axes)

    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    if not is_cbp:
      # Normalize so that all scores sum to one.
      score /= jnp.mean(score) + 1e-9

    return score
  

@gin.configurable
class LayerReset(BaseRecycler):
  """Reset all weights of some layers.

  This class implements the primacy bias paper, in which all weights of the
  last layers of the model are randomly reinitalized.

  Attributes:
  """

  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval

  def update_weights(self, intermediates, params, key, opt_state):
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')
    mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    random_keys_dict = {k: None for k in param_dict}
    # set mask = 1 for the weights that will be recycled
    for param_key, param in param_dict.items():
      param_layer_name = param_key.split('/')[1]
      if param.ndim != 1 and param_layer_name in self.reset_layers:
        mask_dict[param_key] = jnp.ones_like(param)
        key, subkey = random.split(key)
        random_keys_dict[param_key] = subkey
        # reset bias
        bias_key = 'params/' + param_layer_name + '/bias'
        new_bias = jnp.zeros_like(param_dict[bias_key])
        param_dict[bias_key] = new_bias

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    masks = flax.core.freeze(
        flax.traverse_util.unflatten_dict(mask_dict, sep='/')
    )
    random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(random_keys_dict, sep='/')
    )
    # reset weights
    weight_random_reset_fn = jax.jit(
        functools.partial(jax.tree_util.tree_map, weight_reinit_random)
    )
    params = weight_random_reset_fn(params, masks, random_keys)

    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reset_momentum))
    new_mu = reset_momentum_fn(opt_state[0][1], masks)
    new_nu = reset_momentum_fn(opt_state[0][2], masks)
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state


@gin.configurable
class NeuronRecycler(BaseRecycler):
  """Recycle the weights connected to dead neurons.

  In convolutional neural networks, we consider a feature map as neuron.

  Attributes:
    next_layers: dict key a current layer name, value next layer name.
    init_method_outgoing: method to init outgoing weights (random, zero).
    weight_scaling: if true, scale reinit weights.
    incoming_scale: scalar for incoming weights.
    outgoing_scale: scalar for outgoing weights.
  """

  def __init__(
      self,
      all_layers_names,
      init_method_outgoing='zero',
      weight_scaling=False,
      incoming_scale=1.0,
      outgoing_scale=1.0,
      network='nature',
      prune_dormant_neurons=False,
      **kwargs,
  ):
    super(NeuronRecycler, self).__init__(all_layers_names, **kwargs)
    self.init_method_outgoing = init_method_outgoing
    self.weight_scaling = weight_scaling
    self.incoming_scale = incoming_scale
    self.outgoing_scale = outgoing_scale
    self.prune_dormant_neurons = prune_dormant_neurons
    self.first_time_pruning = True
    # prepare a dict that has pointer to next layer give a layer name
    # this is needed because neuron recycle reinitalizes both sides
    # (incoming and outgoing weights) of a neuron and needs a point to the
    # outgoing weights.
    self.next_layers = {}
    for current_layer, next_layer in zip(
        all_layers_names[:-1], all_layers_names[1:]
    ):
      self.next_layers[current_layer] = next_layer

    # we don't recycle the neurons in the output layer.
    self.reset_layers = self.reset_layers[:-1]

    # if network is resnet, recycle only the incoming/outgoing of the first conv
    # layer in each block and final dense layer
    if network == 'resnet':
      self.reset_layers = []
      for layer in self.all_layers_names:
        if 'Conv_1' in layer or 'Conv_3' in layer or 'Dense' in layer:
          self.reset_layers.append(layer)

  def intersected_dead_neurons_with_last_reset(
      self, intermediates, update_step
  ):
    if self.is_reset(update_step):
      log_dict = self.log_intersected_dead_neurons(intermediates, update_step)
      self.log_historical_dead_neuron_overlapping(intermediates, update_step)
      return log_dict
    else:
      return None

  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval

  def is_intermediated_required(self, update_step):
    is_logging = self.is_logging_step(update_step)
    is_update_iter = self.is_update_iter(update_step)
    return is_logging or is_update_iter

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]
    self.reset_layers = self.reset_layers[:-1]

  def update_weights(self, intermediates, params, key, opt_state):
    if self.prune_dormant_neurons:
      new_param = self.prune_dead_neurons(
          intermediates, params, key, opt_state
      )
    else:
      new_param, opt_state = self.recycle_dead_neurons(
          intermediates, params, key, opt_state
      )
    return new_param, opt_state

  def recycle_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        intermedieates, sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')

    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        incoming_mask_dict,
        outgoing_mask_dict,
        incoming_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    ) = self.create_masks(param_dict, activations_score_dict, key)

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    incoming_random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_random_keys_dict, sep='/')
    )
    if self.init_method_outgoing == 'random':
      outgoing_random_keys = flax.core.freeze(
          flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/')
      )
    # reset incoming weights
    incoming_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_mask_dict, sep='/')
    )
    reinit_fn = functools.partial(
        weight_reinit_random,
        weight_scaling=self.weight_scaling,
        scale=self.incoming_scale,
        weights_type='incoming',
    )
    weight_random_reset_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reinit_fn))
    params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)

    # reset outgoing weights
    outgoing_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/')
    )
    # print(outgoing_mask.__class__) # <class 'flax.core.frozen_dict.FrozenDict'>
    # for k in outgoing_mask.keys():
    #   print(11, k)
    #   for k1 in outgoing_mask[k]:
    #     print(22, k1)
    #     for k2 in outgoing_mask[k][k1]:
    #       print(33, k2)
    #       if k2 == 'kernel':
    #         print(55, outgoing_mask[k][k1][k2].shape)
    # 11 params
    # 22 Conv_0
    # 33 bias # None
    # 33 kernel
    # 55 (8, 8, 4, 32)
    # 22 Conv_1
    # 33 bias
    # 33 kernel
    # 55 (4, 4, 32, 64)
    # 22 Conv_2
    # 33 bias
    # 33 kernel
    # 55 (3, 3, 64, 64)
    # 22 Dense_0
    # 33 bias
    # 33 kernel
    # 55 (7744, 512)
    # 22 final_layer
    # 33 bias
    # 33 kernel
    # 55 (512, 6)
    if self.init_method_outgoing == 'random':
      reinit_fn = functools.partial(
          weight_reinit_random,
          weight_scaling=self.weight_scaling,
          scale=self.outgoing_scale,
          weights_type='outgoing',
      )
      weight_random_reset_fn = jax.jit(
          functools.partial(jax.tree_util.tree_map, reinit_fn)
      )
      params = weight_random_reset_fn(
          params, outgoing_mask, outgoing_random_keys
      )
    elif self.init_method_outgoing == 'zero':
      weight_zero_reset_fn = jax.jit(
          functools.partial(jax.tree_util.tree_map, weight_reinit_zero)
      )
      params = weight_zero_reset_fn(params, outgoing_mask)
    else:
      raise ValueError(f'Invalid init method: {self.init_method_outgoing}')
    
    def has_dormant_neurons(mask):
      for k in outgoing_mask.keys():
        for k1 in outgoing_mask[k]:
          for k2 in outgoing_mask[k][k1]:
            if k2 == 'kernel':
              has_dormant_neurons = jnp.count_nonzero(outgoing_mask[k][k1][k2]) > 0
              if has_dormant_neurons:
                return True
      return False
    # if self.first_time_pruning and has_dormant_neurons(outgoing_mask):
    #   self.outgoing_mask = outgoing_mask
    #   self.first_time_pruning = False
    #   params_norm = jnp.where(self.outgoing_mask == 1, new_param, param)

    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reset_momentum))
    # incoming_mask = flax.core.frozen_dict.unfreeze(incoming_mask)
    new_mu = reset_momentum_fn(opt_state[0][1], flax.core.frozen_dict.unfreeze(incoming_mask))
    new_mu = reset_momentum_fn(new_mu, flax.core.frozen_dict.unfreeze(outgoing_mask))
    new_nu = reset_momentum_fn(opt_state[0][2], flax.core.frozen_dict.unfreeze(incoming_mask))
    new_nu = reset_momentum_fn(new_nu, flax.core.frozen_dict.unfreeze(outgoing_mask))
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state
  
  def prune_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        intermedieates, sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')

    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        incoming_mask_dict,
        outgoing_mask_dict,
        incoming_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    ) = self.create_masks(param_dict, activations_score_dict, key)

    # params = flax.core.freeze(
    #     flax.traverse_util.unflatten_dict(param_dict, sep='/')
    # )
    # if self.init_method_outgoing == 'random':
    #   outgoing_random_keys = flax.core.freeze(
    #       flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/')
    #   )

    # # reset outgoing weights
    # outgoing_mask = flax.core.freeze(
    #     flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/')
    # )
    # if self.init_method_outgoing == 'random':
    #   reinit_fn = functools.partial(
    #       weight_reinit_random,
    #       weight_scaling=self.weight_scaling,
    #       scale=self.outgoing_scale,
    #       weights_type='outgoing',
    #   )
    #   weight_random_reset_fn = jax.jit(
    #       functools.partial(jax.tree_util.tree_map, reinit_fn)
    #   )
    #   params = weight_random_reset_fn(
    #       params, outgoing_mask, outgoing_random_keys
    #   )
    # elif self.init_method_outgoing == 'zero':
    #   weight_zero_reset_fn = jax.jit(
    #       functools.partial(jax.tree_util.tree_map, weight_reinit_zero)
    #   )
    #   params = weight_zero_reset_fn(params, outgoing_mask)
    # else:
    #   raise ValueError(f'Invalid init method: {self.init_method_outgoing}')
    return params

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_neurons_threshold

  def create_masks(self, param_dict, activations_dict, key):
    """create the masks for recycled weights based on neurons scores.

    Args:
      param_dict: dict of model params.
      activations_dict: dict of the neuron score of each layer.
      key: used seed for random weights.

    Returns:
      incoming_mask_dict
      outgoing_mask_dict
      ingoing_random_keys_dict
      outgoing_random_keys_dict
      param_dict
    """
    incoming_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    outgoing_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    ingoing_random_keys_dict = {k: None for k in param_dict}
    outgoing_random_keys_dict = (
        {k: None for k in param_dict}
        if self.init_method_outgoing == 'random'
        else {}
    )
    # prepare mask of incoming and outgoing recycled connections
    for k in self.reset_layers:
      param_key = 'params/' + k + '/kernel'
      param = param_dict[param_key]
      # This won't work for DRQ, since returned keys can be a list.
      # We don't support that at the moment.
      next_key = self.next_layers[k]
      if isinstance(next_key, list):
        next_key = next_key[0]
      next_param = param_dict['params/' + next_key + '/kernel']
      activation = activations_dict[k + '_act/__call__'][0]
      # TODO(evcu) Maybe use per_layer random keys here.
      neuron_mask = self._score2mask(activation, param, next_param, key)

      # the for loop handles the case where a layer has multiple next layers
      # like the case in DrQ where the output layer has multihead.
      next_keys = (
          self.next_layers[k]
          if isinstance(self.next_layers[k], list)
          else [self.next_layers[k]]
      )
      for next_k in next_keys:
        next_param_key = 'params/' + next_k + '/kernel'
        next_param = param_dict[next_param_key]
        incoming_mask, outgoing_mask = self.create_mask_helper(
            neuron_mask, param, next_param
        )
        incoming_mask_dict[param_key] = incoming_mask
        outgoing_mask_dict[next_param_key] = outgoing_mask
        key, subkey = random.split(key)
        ingoing_random_keys_dict[param_key] = subkey
        if self.init_method_outgoing == 'random':
          key, subkey = random.split(key)
          outgoing_random_keys_dict[next_param_key] = subkey

        if self.prune_dormant_neurons: # NOTE (ZW) stop the gradients flowing through dormant neurons
          # NOTE (ZW) Log the magnitude of outgoing weights of dormant neurons
          print('Pruning {} outgoing weights at layer {}'.format(outgoing_mask.sum(), k))
          # next_param = jnp.where(~outgoing_mask, next_param, jax.lax.stop_gradient(next_param))
        
      #     if self.first_time_pruning and (jnp.count_nonzero(outgoing_mask) > 0):
      #       self.next_param_key = next_param_key
      #       self.outgoing_mask = outgoing_mask
      #       self.first_time_pruning = False

      # if (not self.first_time_pruning) and self.prune_dormant_neurons:
      #   print(jnp.count_nonzero(self.outgoing_mask), self.next_param_key)
      #   frozen_params = jnp.where(self.outgoing_mask == 1, jnp.zeros_like(param_dict[self.next_param_key]), param_dict[self.next_param_key])
      #   print(jnp.linalg.vector_norm(frozen_params).item())
      #   import time
      #   time.sleep(2)
      #   if config['use_wandb']:
      #     wandb.log({'frozen_params_norm': jnp.linalg.vector_norm(frozen_params).item(), 'grad_step': self._last_update_step})

      # reset bias
      bias_key = 'params/' + k + '/bias'
      new_bias = jnp.zeros_like(param_dict[bias_key])
      if self.prune_dormant_neurons:
        new_bias -= 99999999
      param_dict[bias_key] = jnp.where(
          neuron_mask, new_bias, param_dict[bias_key]
      ) # True entities in param_dict[bias_key] will be replaced by new_bias

    return (
        incoming_mask_dict,
        outgoing_mask_dict,
        ingoing_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    )

  def create_mask_helper(self, neuron_mask, current_param, next_param):
    """generate incoming and outgoing weight mask given dead neurons mask.

    Args:
      neuron_mask: mask of size equals the width of a layer.
      current_param: incoming weights of a layer.
      next_param: outgoing weights of a layer.

    Returns:
      incoming_mask
      outgoing_mask
    """

    def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
      """create a mask of weight matrix given 1D vector of neurons mask.

      Args:
        expansion_axis: List contains 1 axis. The dimension to expand the mask
          for dense layers (weight shape 2D).
        expansion_axes: List conrtains 3 axes. The dimensions to expand the
          score for convolutional layers (weight shape 4D).
        param: weight.
        neuron_mask: 1D mask that represents dead neurons(features).

      Returns:
        mask: mask of weight.
      """
      if param.ndim == 2:
        axes = expansion_axis
        # flatten layer
        # The size of neuron_mask is the same as the width of last conv layer.
        # This conv layer will be flatten and connected to dense layer.
        # we repeat each value of a feature map to cover the spatial dimension.
        if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
          num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
          neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
      elif param.ndim == 4:
        axes = expansion_axes
      mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
      for i in range(len(axes)):
        mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
      return mask

    incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
    outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
    return incoming_mask, outgoing_mask

  # def estimate_all_neuron_score(self, param_dict, activations_dict):
  #   for k in self.reset_layers:
  #     param_key = 'params/' + k + '/kernel'
  #     param = param_dict[param_key]
  #     # This won't work for DRQ, since returned keys can be a list.
  #     # We don't support that at the moment.
  #     next_key = self.next_layers[k]
  #     if isinstance(next_key, list):
  #       next_key = next_key[0]
  #     next_param = param_dict['params/' + next_key + '/kernel']
  #     activation = activations_dict[k + '_act/__call__'][0]
  #     # TODO(evcu) Maybe use per_layer random keys here.
  #     score = self.estimate_neuron_score(activation)

@gin.configurable
class NeuronRecyclerScheduled(NeuronRecycler):
  """Fixed scheduled version of the NeuronRecycler."""

  def __init__(
      self,
      *args,
      score_type='redo',
      recycle_rate=0.3,
      **kwargs,
  ):
    super(NeuronRecyclerScheduled, self).__init__(*args, **kwargs)
    self.score_type = score_type
    self.recycle_rate = recycle_rate

  def _score2mask(self, activation, param, next_param, key):
    is_cbp = self.score_type == 'cbp'
    score = self.estimate_neuron_score(activation, is_cbp=is_cbp)
    if self.score_type == 'redo':
      pass
    elif self.score_type == 'random':
      new_key = random.fold_in(key, self._last_update_step)
      score = random.permutation(new_key, score, independent=True)
    elif self.score_type == 'redo_inverted':
      score = -score
    # Metric used in Continual Backprop pape.
    elif self.score_type == 'cbp':
      next_axes = list(range(param.ndim))
      del next_axes[-2]
      current_axes = list(range(param.ndim))
      del current_axes[-1]
      if next_param.ndim == 2 and param.ndim == 4:
        new_shape = activation.shape[1:] + (-1,)
        next_param = jnp.reshape(next_param, new_shape)
      score *= jnp.sum(jnp.abs(next_param), axis=next_axes) / jnp.sum(
          jnp.abs(param), axis=current_axes
      )
    multiplier = max(0, self._last_update_step / self.reset_end_step)
    ones_fraction = float(jnp.cos(jnp.pi * 0.5 * multiplier))
    ones_fraction *= self.recycle_rate
    logging.info(
        'score_type: %s, multiplier: %f, ones_fraction=%f',
        self.score_type,
        multiplier,
        ones_fraction,
    )
    return leastk_mask(score, ones_fraction)


class PrunerDontRecycle(BaseRecycler):
  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval
  
  def maybe_update_weights(
      self, update_step, intermediates, params, key, opt_state
  ):
    self._last_update_step = update_step
    return params, opt_state