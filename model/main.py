import json
import os 
import pickle
import numpy as np
from datetime import datetime
import jax

from waymax import config as _config

from rnnbc import make_train

##
# CONFIG
##

# Training config
config = {
    'anneal_lr': False,
    'bins': 128,
    'discrete': False,
    'freq_save': 1,
    'key': 42,
    'lr': 3e-4,
    "max_grad_norm": 0.5,
    'max_num_obj': 8,
    'max_num_rg_points': 20000,
    'num_envs': 32,
    'num_envs_eval': 16,
    "num_epochs": 100,
    'num_steps': 80,
    'roadgraph_top_k': 100,
    'shuffle_seed': 123,
    'shuffle_buffer_size': 1_000,
    'total_timesteps': 100,
    'training_path': '/data/saruman/cleain/WOD_1_1_0/tf_example/training/training_tfexample.tfrecord@1000',
    'validation_path': '/data/saruman/cleain/WOD_1_1_0/tf_example/validation/validation_tfexample.tfrecord@150'
    }

# Ckeckpoint path
current_time = datetime.now() 
date_string = current_time.strftime("%Y%m%d_%H%M%S")

log_folder = f"logs/{date_string}"
os.makedirs(log_folder, exist_ok='True')

config['log_folder'] = log_folder

# Save training config
training_args = config

with open(os.path.join(log_folder, 'args.json'), 'w') as json_file:
    json.dump(training_args, json_file)

# Data iter config
WOD_1_1_0_TRAINING = _config.DatasetConfig(
    path=config['training_path'],
    max_num_rg_points=config['max_num_rg_points'],
    shuffle_seed=config['shuffle_seed'],
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['num_envs'],),
    max_num_objects=config['max_num_obj'],
    repeat=1
)

# Data iter config
WOD_1_1_0_VALIDATION = _config.DatasetConfig(
    path=config['validation_path'],
    max_num_rg_points=config['max_num_rg_points'],
    shuffle_seed=None,
    data_format=_config.DataFormat.TFRECORD,
    batch_dims = (config['num_envs_eval'],),
    max_num_objects=config['max_num_obj'],
    repeat=1
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
                      WOD_1_1_0_TRAINING,
                      WOD_1_1_0_VALIDATION)

# with jax.disable_jit(): # DEBUG
training_dict = training.train()
