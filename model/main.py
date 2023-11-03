import jax
from jax import random

from waymax import config as _config
from waymax import dataloader

from rnnbc import make_train

# Training config
config = {
    'ANNEAL_LR': False,
    'bins': 128,
    'discrete': False,
    'TOTAL_TIMESTEPS': 100,
    'roadgraph_top_k': 100,
    'LR': 3e-4,
    'NUM_ENVS': 1,
    "NUM_EPOCHS": 1,
    'NUM_STEPS': 10,
    "n_train_per_epoch": 2,
    "MAX_GRAD_NORM": 0.5,
    'KEY': random.PRNGKey(42),
    'max_num_obj': 8
    }

# Data iter config
WOD_1_1_0_TRAINING = _config.DatasetConfig(
    path='/data/saruman/cleain/WOD_1_1_0/tf_example/training/training_tfexample.tfrecord@1000',
    max_num_rg_points=20000,
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['NUM_ENVS'],),
    max_num_objects=config['max_num_obj']
)

# Env config
env_config = _config.EnvironmentConfig(
    controlled_object=_config.ObjectType.SDC,
    max_num_objects=config['max_num_obj']
)

# Training

train_fn = make_train(config,
                      env_config,
                      WOD_1_1_0_TRAINING)

# jit_train_fn = jax.jit(train_fn)
_, metrics = train_fn()