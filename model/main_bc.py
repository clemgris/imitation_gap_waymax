from datetime import datetime
import functools
import jax
import jax.numpy as jnp
import json
import os

from waymax import config as _config
from waymax import dataloader
from rnnbc import make_train

from dataset.config import HEADING_RADIUS

from utils.dataloader import tf_examples_dataset
# Desable preallocation for jax and tensorflow
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

##
# CONFIG
##

# Training config
config = {
    'bins': 128,
    'discrete': False,
    'dynamics': 'delta',
    'extractor': 'ExtractObs',
    'feature_extractor': 'KeyExtractor',
    'feature_extractor_kwargs': {'final_hidden_layers': 128,
                                #  'hidden_layers': {'roadgraph_map': 8},
                                 'keys': ['xy',
                                        #   'proxy_goal',
                                        'heading',
                                        #   'roadgraph_map'
                                          ]},
    'freq_eval': 10,
    'freq_save': 10,
    'include_sdc_paths': False,
    'key': 42,
    'lr': 3e-4,
    'lr_scheduler': False,
    "max_grad_norm": 0.5,
    'max_num_obj': 8,
    'max_num_rg_points': 20000,
    'num_envs': 16,
    'num_envs_eval': 16,
    "num_epochs": 500,
    'num_steps': 80,
    'obs_mask': None, # 'SpeedGaussianNoise', #'SpeedConicObsMask',
    'obs_mask_kwargs': None,
        
        # {
        # 'v_max': 15,
        # 'sigma_max':5
        # },
        
        # {
        # 'radius': 100, # Sanity check (as full obs)
        # 'angle_min': 2 * jnp.pi, # Sanity check (as full obs)
        # 'v_max': 15, # 15,
        # },
    'roadgraph_top_k': 2000,
    'shuffle_seed': 123,
    'shuffle_buffer_size': 1000, # 1000
    'total_timesteps': 100,
    'min_mean_speed': None,
    'num_files': 100,
    'training_path': '/data/saruman/shared/WOD_1_1_0/tf_example/training/training_tfexample.tfrecord@1000',
    'validation_path': '/data/saruman/shared/WOD_1_1_0/tf_example/validation/validation_tfexample.tfrecord@150',
    'should_cache': True
    }

# for radius in [20, 40, 60, 80, 100]:
#     for angle_min in [jnp.pi / 8, jnp.pi / 4, jnp.pi / 2, jnp.pi, jnp.pi * 2]:
        
# config['obs_mask_kwargs']['radius'] = radius
# config['obs_mask_kwargs']['angle_min'] = angle_min

# for sigma_max in [0,1,3,5]:
#     
    # config['obs_mask_kwargs']['sigma_max'] = sigma_max

# Ckeckpoint path
current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs/{date_string}"
os.makedirs(log_folder, exist_ok='True')

config['log_folder'] = log_folder
if 'heading' in config['feature_extractor_kwargs']['keys']:
    config['HEADING_RADIUS'] = HEADING_RADIUS
    
# Save training config
training_args = config

with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
    json.dump(training_args, json_file, indent=4)

# Data iter config
WOD_1_1_0_TRAINING = _config.DatasetConfig(
    path=config['training_path'],
    max_num_rg_points=config['max_num_rg_points'],
    shuffle_seed=config['shuffle_seed'],
    shuffle_buffer_size=config['shuffle_buffer_size'],
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['num_envs'],),
    max_num_objects=config['max_num_obj'],
    include_sdc_paths=config['include_sdc_paths'],
    repeat=None
)

# Data iter config
WOD_1_1_0_VALIDATION = _config.DatasetConfig(
    path=config['validation_path'],
    max_num_rg_points=config['max_num_rg_points'],
    shuffle_seed=None,
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['num_envs_eval'],),
    max_num_objects=config['max_num_obj'],
    include_sdc_paths=config['include_sdc_paths'],
    repeat=1
)

# Training dataset
train_dataset = tf_examples_dataset(
    path=WOD_1_1_0_TRAINING.path,
    data_format=WOD_1_1_0_TRAINING.data_format,
    preprocess_fn=functools.partial(dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_TRAINING),
    shuffle_seed=WOD_1_1_0_TRAINING.shuffle_seed,
    shuffle_buffer_size=WOD_1_1_0_TRAINING.shuffle_buffer_size,
    repeat=WOD_1_1_0_TRAINING.repeat,
    batch_dims=WOD_1_1_0_TRAINING.batch_dims,
    num_shards=WOD_1_1_0_TRAINING.num_shards,
    deterministic=WOD_1_1_0_TRAINING.deterministic,
    drop_remainder=WOD_1_1_0_TRAINING.drop_remainder,
    tf_data_service_address=WOD_1_1_0_TRAINING.tf_data_service_address,
    batch_by_scenario=WOD_1_1_0_TRAINING.batch_by_scenario,
    filter_function=None,
    num_files = config['num_files'],
    should_cache = config['should_cache']
)

data = train_dataset.as_numpy_iterator().next() # DEBUG

# Validation dataset
val_dataset = dataloader.tf_examples_dataset(
    path=WOD_1_1_0_VALIDATION.path,
    data_format=WOD_1_1_0_VALIDATION.data_format,
    preprocess_fn=functools.partial(dataloader.preprocess_serialized_womd_data, config=WOD_1_1_0_VALIDATION),
    shuffle_seed=WOD_1_1_0_VALIDATION.shuffle_seed,
    shuffle_buffer_size=WOD_1_1_0_VALIDATION.shuffle_buffer_size,
    repeat=WOD_1_1_0_VALIDATION.repeat,
    batch_dims=WOD_1_1_0_VALIDATION.batch_dims,
    num_shards=WOD_1_1_0_VALIDATION.num_shards,
    deterministic=WOD_1_1_0_VALIDATION.deterministic,
    drop_remainder=WOD_1_1_0_VALIDATION.drop_remainder,
    tf_data_service_address=WOD_1_1_0_VALIDATION.tf_data_service_address,
    batch_by_scenario=WOD_1_1_0_VALIDATION.batch_by_scenario,
)

# Env config
env_config = _config.EnvironmentConfig(
    controlled_object=_config.ObjectType.SDC,
    max_num_objects=config['max_num_obj']
)

##
# TRAINING
##
training = make_train(config,
                    env_config,
                    train_dataset,
                    val_dataset,
                    data # DEBUG
                    )

with jax.disable_jit(): # DEBUG
    training_dict = training.train()
    