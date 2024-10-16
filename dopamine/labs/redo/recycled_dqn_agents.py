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
"""Variant of JaxDQN for recycling."""

import functools

import flax.core
from dopamine.jax import losses
from dopamine.jax.agents.dqn import dqn_agent
from dopamine.labs.redo import networks
from dopamine.labs.redo import weight_recyclers
from dopamine.metrics import statistics_instance
import gin
import jax
import jax.numpy as jnp
import optax
import flax


@functools.partial(jax.jit, static_argnames=['apply_fn'])
def loss_fn(params, target, state, action, apply_fn):
  """Calculates the loss."""

  def q_online(state):
    return apply_fn(params, state)

  q_values = jax.vmap(q_online)(state).q_values
  q_values = jnp.squeeze(q_values)
  replay_chosen_q = jax.vmap(lambda x, y: x[y])(q_values, action)
  loss = jnp.mean(jax.vmap(losses.huber_loss)(target, replay_chosen_q))
  # TODO(all) add weight decay. See jax.example_libraries.optimizers.l2_norm
  return loss


@functools.partial(jax.jit, static_argnames=['apply_fn', 'cumulative_gamma'])
def get_gradients(
    online_params, apply_fn, target_params, batch, cumulative_gamma
):
  """Calculates the gradiens and the loss."""

  def q_target(state):
    return apply_fn(target_params, state)

  q_vals = jax.vmap(q_target)(batch['next_state']).q_values
  q_vals = jnp.squeeze(q_vals)
  replay_next_qt_max = jnp.max(q_vals, 1)
  # Calculate the Bellman target value.
  #   Q_t = R_t + \gamma^N * Q'_t+1
  # where,
  #   Q'_t+1 = \argmax_a Q(S_t+1, a)
  #          (or) 0 if S_t is a terminal state,
  # and
  #   N is the update horizon (by default, N=1).
  target = jax.lax.stop_gradient(
      batch['reward']
      + cumulative_gamma * replay_next_qt_max * (1.0 - batch['terminal'])
  )
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grad = grad_fn(
      online_params,
      target=target,
      state=batch['state'],
      action=batch['action'],
      apply_fn=apply_fn,
  )
  return loss, grad


@functools.partial(jax.jit, static_argnames=['optimizer'])
def apply_updates_jitted(online_params, grad, optimizer_state, optimizer):
  # print(grad.__class__, optimizer_state.__class__, online_params.__class__)
  updates, optimizer_state = optimizer.update(
      flax.core.unfreeze(grad), optimizer_state, params=flax.core.unfreeze(online_params)
  )
  # print(grad.__class__, optimizer_state.__class__, updates.__class__)
  new_online_params = optax.apply_updates(flax.core.unfreeze(online_params), updates)
  return new_online_params, optimizer_state


@gin.configurable
class RecycledDQNAgent(dqn_agent.JaxDQNAgent):
  """DQN Agent with masked parameters."""

  def __init__(
      self,
      num_actions,
      network='resnet',
      reset_mode=None,
      width=1,
      num_updates_per_train_step=1,
      batch_size_statistics=256,
      target_update_strategy='training_step',
      weight_decay=0.0,
      summary_writer=None,
      is_debugging=False,
  ):
    network_name = network
    if network == 'resnet':
      network = networks.ScalableDQNResNet
    elif network == 'nature':
      network = networks.ScalableNatureDQNNetwork
    elif network == 'nature_with_one_extra_ffn':
      network = networks.ScalableNatureDQNNetworkWithOneExtraFFN
    else:
      raise ValueError(f'Invalid network: {network}')
    super().__init__(
        num_actions,
        network=functools.partial(network, width=width),
        summary_writer=summary_writer,
    )
    self.network = network

    if weight_decay > 0:
      # TODO(gsokar) we may compare the performance with adamw.
      # lets keep it under a condition till we check its effect.
      self.optimizer = optax.chain(
          optax.add_decayed_weights(weight_decay), self.optimizer
      )
      self.optimizer_state = self.optimizer.init(self.online_params)
    # NOTE (ZW) parameter dict structure
    # for k in self.online_params.keys():
    #   print(11, k)
    #   for k1 in self.online_params[k].keys():
    #     print(22, k1)
    #     for k2 in self.online_params[k][k1].keys():
    #       print(33, k2, self.online_params[k][k1][k2].shape)
        # 11 params
        # 22 Conv_0
        # 33 kernel (8, 8, 4, 32)
        # 33 bias (32,)
        # 22 Conv_1
        # 33 kernel (4, 4, 32, 64)
        # 33 bias (64,)
        # 22 Conv_2
        # 33 kernel (3, 3, 64, 64)
        # 33 bias (64,)
        # 22 Dense_0
        # 33 kernel (7744, 512)
        # 33 bias (512,)
        # 22 final_layer
        # 33 kernel (512, 6)
        # 33 bias (6,)

    self.batch_size_statistics = batch_size_statistics
    self.target_update_strategy = target_update_strategy
    self.is_debugging = is_debugging
    if reset_mode:
      if reset_mode == 'neurons':
        self.weight_recycler = weight_recyclers.NeuronRecycler(
            network.layer_names, network=network_name
        )
      elif reset_mode == 'neurons_scheduled':
        self.weight_recycler = weight_recyclers.NeuronRecyclerScheduled(
            network.layer_names, network=network_name
        )
      elif reset_mode == 'weights':
        self.weight_recycler = weight_recyclers.LayerReset(network.layer_names)
      else:
        raise ValueError(f'Invalid reset mode: {reset_mode}')
    else:
      self.weight_recycler = weight_recyclers.BaseRecycler(network.layer_names)
    self._num_updates_per_train_step = num_updates_per_train_step

  def _log_stats(self, log_dict, step):
    if log_dict is None:
      return
    stats = []
    for k, v in log_dict.items():
      stats.append(statistics_instance.StatisticsInstance(k, v, step=step))
    self.collector_dispatcher.write(
        stats, collector_allowlist=self._collector_allowlist
    )

  def _train_step(self):
    if self._replay.add_count > self.min_replay_history:
      if self.training_steps % self.update_period == 0:
        for step in range(self._num_updates_per_train_step):
          self.gradient_step = (
              (self.training_steps // self.update_period)
              * self._num_updates_per_train_step
          ) + step
          self._training_step_update()
          if self.target_update_strategy == 'update_step':
            if self.gradient_step % self.target_update_period == 0:
              self._sync_weights()

      # The original agent updates target based on training steps. We need to
      # analyze whether we need to change it to gradient_step in case of
      # high replay ratio (i.e., _num_updates_per_train_step > 1) and recycling.
      if self.target_update_strategy == 'training_step':
        if self.training_steps % self.target_update_period == 0:
          self._sync_weights()

    self.training_steps += 1

  def _training_step_update(self):
    # We are using # gradient_update_steps in our calculations and logging.
    update_step = self.gradient_step
    is_logging = (
        update_step > 0 and update_step % self.summary_writing_frequency == 0
    )

    self._sample_from_replay_buffer()
    batch = {
        k: self.preprocess_fn(v) if k in ['state', 'next_state'] else v
        for k, v in self.replay_elements.items()
    }
    online_params = self.online_params

    loss, grad = get_gradients(
        online_params=online_params,
        apply_fn=self.network_def.apply,
        target_params=self.target_network_params,
        batch=batch,
        cumulative_gamma=self.cumulative_gamma,
    )
    new_online_params, self.optimizer_state = apply_updates_jitted(
        online_params, grad, self.optimizer_state, self.optimizer
    )
    self.online_params = new_online_params
    online_params = self.online_params

    if is_logging:
      self._log_stats({'HuberLoss': float(loss)}, update_step)

    # Neuron/layer recycling starts if reset_mode is not None.
    # Otherwise, we log dead neurons over training for standard agent.
    is_intermediated = self.weight_recycler.is_intermediated_required(
        update_step
    )
    # get intermediate activation per layer to calculate neuron score.
    intermediates = (
        self.get_intermediates(online_params) if is_intermediated else None
    )
    log_dict_neurons = self.weight_recycler.maybe_log_deadneurons(
        update_step, intermediates
    )
    # logging dead neurons.
    self._log_stats(log_dict_neurons, update_step)
    if self.is_debugging:
      log_dict_intersected = (
          self.weight_recycler.intersected_dead_neurons_with_last_reset(
              intermediates, update_step
          )
      )
      self._log_stats(log_dict_intersected, update_step)

    # Neuron/layer recyling.
    self._rng, key = jax.random.split(self._rng)
    online_params, opt_state = self.weight_recycler.maybe_update_weights(
        update_step, intermediates, online_params, key, self.optimizer_state
    )
    self.optimizer_state = opt_state
    self.online_params = online_params

  def _sample_batch_for_statistics(self):
    samples = self._replay.sample_transition_batch(
        batch_size=self.batch_size_statistics
    )
    types = self._replay.get_transition_elements()
    for element, element_type in zip(samples, types):
      if element_type.name == 'state':
        states = self.preprocess_fn(element)
        break
    return states

  def get_intermediates(self, online_params):
    batch = self._sample_batch_for_statistics()

    def apply_data(x):
      filter_rep = lambda l, _: l.name is not None and 'act' in l.name
      return self.network_def.apply(
          online_params,
          x,
          capture_intermediates=filter_rep,
          mutable=['intermediates'],
      )

    _, state = jax.vmap(apply_data)(batch)
    return state['intermediates']


# NOTE (ZW) Added
from dopamine.labs.redo import sparse_util

@functools.partial(jax.jit, static_argnames=['optimizer', 'is_prune'])
def apply_normal_updates_jitted(online_params, grad, optimizer_state, optimizer, is_prune):
  updates, optimizer_state = optimizer.update(
      flax.core.unfreeze(grad), optimizer_state, params=flax.core.unfreeze(online_params), is_prune=is_prune
  )
  new_online_params = optax.apply_updates(flax.core.unfreeze(online_params), updates)
  return new_online_params, optimizer_state

# @functools.partial(jax.jit, static_argnames=['optimizer'])
def apply_reset_updates(online_params, grad, optimizer_state, optimizer, is_prune):
  updates, optimizer_state = optimizer.update(
      flax.core.unfreeze(grad), optimizer_state, params=flax.core.unfreeze(online_params), is_prune=is_prune
  )
  new_online_params = optax.apply_updates(flax.core.unfreeze(online_params), updates)
  # new_optimizer_state = update_fn(updates, new_optimizer_state, new_online_params)
  return new_online_params, optimizer_state


class PrunnerDQNAgent(RecycledDQNAgent):

  def __init__(
      self,
      **kwargs
  ):
    super(PrunnerDQNAgent, self).__init__(**kwargs)
    self.weight_recycler = weight_recyclers.PrunerDontRecycle(self.network.layer_names)

  def _build_networks_and_optimizer(self):
    self._rng, init_rng, updater_rng = jax.random.split(self._rng, num=3)
    self.online_params = self.network_def.init(init_rng, x=self.state)
    self.optimizer = dqn_agent.create_optimizer(self._optimizer_name)
    self.updater = sparse_util.create_updater_from_config(rng_seed=updater_rng)
    self.post_gradient_update = jax.jit(self.updater.post_gradient_update)
    self.pre_forward_update = self.updater.pre_forward_update #jax.jit(self.updater.pre_forward_update)
    self.optimizer = self.updater.wrap_optax(self.optimizer)
    self.optimizer_state = self.optimizer.init(self.online_params)
    self.target_network_params = self.online_params

  def _training_step_update(self):
    # We are using # gradient_update_steps in our calculations and logging.
    update_step = self.gradient_step

    self._sample_from_replay_buffer()
    batch = {
        k: self.preprocess_fn(v) if k in ['state', 'next_state'] else v
        for k, v in self.replay_elements.items()
    }
    online_params = self.online_params

    # Neuron/layer recycling starts if reset_mode is not None.
    # Otherwise, we log dead neurons over training for standard agent.
    is_intermediated = self.weight_recycler.is_intermediated_required(
        update_step
    )
    # get intermediate activation per layer to calculate neuron score.
    intermediates = (
        self.get_intermediates(online_params) if is_intermediated else None
    )
    log_dict_neurons = self.weight_recycler.maybe_log_deadneurons(
        update_step, intermediates
    )
    # logging dead neurons.
    self._log_stats(log_dict_neurons, update_step)
    if self.is_debugging:
      log_dict_intersected = (
          self.weight_recycler.intersected_dead_neurons_with_last_reset(
              intermediates, update_step
          )
      )
      self._log_stats(log_dict_intersected, update_step)
    # NOTE------------------------------------------------------
    is_prune = self.weight_recycler.is_reset(update_step)
    if is_prune:
      intermediates = (
          self.get_intermediates(online_params)
      ) if intermediates is None else intermediates
      activations_score_dict = flax.traverse_util.flatten_dict(
          intermediates, sep='/'
      )
      # pass the activation value to the pruner
      self.pre_forward_update(
          activations_score_dict, self.weight_recycler.reset_layers, self.optimizer_state
      )
    # -----------------------------------------------------------
    loss, grad = get_gradients(
        online_params=online_params,
        apply_fn=self.network_def.apply,
        target_params=self.target_network_params,
        batch=batch,
        cumulative_gamma=self.cumulative_gamma,
    )
    # self.optimizer_state.count + 4983 == update_step
    if is_prune:
      new_online_params, self.optimizer_state = apply_reset_updates(
          online_params, grad, self.optimizer_state, self.optimizer, is_prune=is_prune
      )
    else:
      new_online_params, self.optimizer_state = apply_normal_updates_jitted(
          online_params, grad, self.optimizer_state, self.optimizer, is_prune=None
      )
    # NOTE------------------------------------------------------
    if is_prune:
      new_online_params = self.post_gradient_update(
          new_online_params, self.optimizer_state
      )
    # -----------------------------------------------------------
    self.online_params = new_online_params
    online_params = self.online_params

    # Neuron/layer recyling.
    # self._rng, key = jax.random.split(self._rng)
    # online_params, opt_state = self.weight_recycler.maybe_update_weights(
    #     update_step, intermediates, online_params, key, self.optimizer_state
    # )
    # self.optimizer_state = opt_state
    self.online_params = online_params
