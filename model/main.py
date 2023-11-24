from datetime import datetime
import functools
import json
import os 

from waymax import config as _config
from waymax import dataloader
from rnnbc import make_train

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

##
# CONFIG
##

# Training config
config = {
    'anneal_lr': False,
    'bins': 128,
    'discrete': False,
    'extractor': 'ExtractXYGoal', #ExtractXY
    'feature_extractor': 'FlattenKeyExtractor', #FlattenXYExtractor
    'feature_extractor_kwargs': {'keys': ['xy', 'proxy_goal'],
                                 'hidden_layers': 128}, # {}
    'freq_save': 10,
    'include_sdc_paths': False,
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
    'training_path': '/data/draco/cleain/WOD_1_1_0/tf_example/training/training_tfexample.tfrecord@1000',
    'validation_path': '/data/draco/cleain/WOD_1_1_0/tf_example/validation/validation_tfexample.tfrecord@150'
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
    include_sdc_paths=config['include_sdc_paths'],
    repeat=1
)


# Training dataset
train_dataset = dataloader.tf_examples_dataset(
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
)

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
                      val_dataset)

# with jax.disable_jit(): # DEBUG
training_dict = training.train()
