import argparse
import functools
import jax
import json
import os
import pickle

from waymax import config as _config
from waymax import dataloader
from eval import make_eval

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

parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument('--expe_id', '-expe', type=str, help='Id of the experiment')
parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
parser.add_argument('--IDM', '-IDM', type=int, help='Use IDM for simulated agents')
parser.add_argument('--GIF', '-GIF', type=int, help='Generate GIFs')

if __name__ == "__main__":
    args = parser.parse_args()

    # Training config
    load_folder = '/data/draco/cleain/imitation_gap_waymax/logs'
    expe_num = args.expe_id

    os.makedirs(f'/data/draco/cleain/imitation_gap_waymax/animation/{expe_num}', exist_ok=True)

    with open(os.path.join(load_folder, expe_num, 'args.json'), 'r') as file:
        config = json.load(file)

    config['num_epochs'] = 1
    config['num_envs_eval'] = 1

    n_epochs = args.epochs

    print('Create datasets')

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
    config['IDM'] = bool(args.IDM)
    config['GIF'] = bool(args.GIF)

    env_config = _config.EnvironmentConfig(
        controlled_object=_config.ObjectType.SDC,
        max_num_objects=config['max_num_obj'],
    )

    ##
    # EVALUATION
    ##

    print('Load network parameters')

    with open(os.path.join(load_folder, expe_num, f'params_{n_epochs}.pkl'), 'rb') as file:
        params = pickle.load(file)

    evaluation = make_eval(config,
                        env_config,
                        val_dataset,
                        params)

    # with jax.disable_jit(): # DEBUG
    evaluation_dict = evaluation.train()

    with open(os.path.join(load_folder, expe_num, f"eval_metrics_{n_epochs}_IDM_{config['IDM']}.pkl"), "wb") as pkl_file:
        pickle.dump(evaluation_dict['metrics'], pkl_file)