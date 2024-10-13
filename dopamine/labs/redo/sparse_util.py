"""This file contains apis to use the jax prunder libraries."""
from absl import logging
import gin
from jaxpruner import api
import ml_collections


@gin.configurable
def create_updater_from_config(
    pruner_type='magnitude',
    dist_type='erk',
    update_end_step=100000,
    update_freq=1000,
    update_start_step=20000,
    sparsity=0.95,
    drop_fraction=0.1,
    rng_seed=8,
    sparsity_type = 'dormant_0.025',
):
  """Gin based wrapper around jaxpruner create function."""
  logging.info(
      'Creating create_updater_from_config fn with the following parameters:'
  )
  logging.info('\t pruner_type: %s', pruner_type)
  logging.info('\t dist_type: %s', dist_type)
  logging.info('\t update_end_step: %s', update_end_step)
  logging.info('\t update_freq: %s', update_freq)
  logging.info('\t update_start_step: %s', update_start_step)
  logging.info('\t sparsity: %s', sparsity)
  logging.info('\t drop_fraction: %s', drop_fraction)
  logging.info('\t rng_seed: %s', rng_seed)

  sparsity_config = ml_collections.ConfigDict()
  sparsity_config.algorithm = pruner_type
  sparsity_config.dist_type = dist_type
  sparsity_config.update_end_step = update_end_step
  sparsity_config.update_freq = update_freq
  sparsity_config.update_start_step = update_start_step
  sparsity_config.sparsity = sparsity
  sparsity_config.rng_seed = rng_seed
  # Used only by rigl and set algorithms.
  sparsity_config.drop_fraction = drop_fraction
  # Used only by activation pruning
  sparsity_config.sparsity_type = sparsity_type
  return api.create_updater_from_config(sparsity_config)