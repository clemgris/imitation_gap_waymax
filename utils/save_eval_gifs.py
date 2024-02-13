import functools
import jax
import jax.numpy as jnp
import json
from waymax import config as _config

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

import argparse
import os
import optax
import pickle
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple

from flax.training.train_state import TrainState

from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents

import sys
sys.path.append('./')
sys.path.append('../')

from dataset.config import N_TRAINING, N_VALIDATION, TRAJ_LENGTH
from model.feature_extractor import KeyExtractor
from model.state_preprocessing import ExtractObs
from model.rnn_policy import ActorCriticRNN, ScannedRNN

from viz import plot_observation_with_goal

class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

extractors = {
    'ExtractObs': ExtractObs
}
feature_extractors = {
    'KeyExtractor': KeyExtractor
}


class save_eval:
    
    def __init__(self, 
                 config,
                 env_config,
                 val_dataset,
                 params,
                 num_gifs,
                 save_path):

        self.num_gifs = num_gifs
        self.save_path = save_path

        self.config = config
        self.env_config = env_config

        # Device
        self.devices = jax.devices()
        print(f'Available devices: {self.devices}')

        # Params
        self.params = params

        # Postprocessing function
        self._post_process = functools.partial(
            dataloader.womd_factories.simulator_state_from_womd_dict,
            include_sdc_paths=config['include_sdc_paths'],
              )

        # VALIDATION DATASET
        self.val_dataset = val_dataset
        
        # Random key
        self.key = self.config['key']

        # DEFINE ENV
        if self.config['dynamics'] == 'bicycle':
            self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
        elif self.config['dynamics'] == 'delta':
            self.wrapped_dynamics_model = dynamics.DeltaLocal()
        else:
            raise ValueError('Unknown dynamics')
        
        self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)

        if config['discrete']:
            action_space_dim = self.dynamics_model.action_spec().shape
            self.dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=self.dynamics_model,
                                                                                  bins=config['bins'] * jnp.array(action_space_dim))
        else:
            self.dynamics_model = self.dynamics_model

        assert(not config['discrete']) # /!\ BUG using scan and DiscreteActionWrapper
        
        if config['IDM']:
            sim_actors = [agents.IDMRoutePolicy(
                is_controlled_func=lambda state: 1 -  state.object_metadata.is_sdc
                )]
        else:
            sim_actors = ()
            
        self.env = _env.PlanningAgentEnvironment(
            dynamics_model=self.wrapped_dynamics_model,
            config=env_config,
            sim_agent_actors=sim_actors
            )

        # DEFINE EXPERT AGENT
        self.expert_agent = agents.create_expert_actor(self.dynamics_model)

        # DEFINE EXTRACTOR AND FEATURE_EXTRACTOR
        self.extractor = extractors[self.config['extractor']](self.config)
        self.feature_extractor = feature_extractors[self.config['feature_extractor']]
        self.feature_extractor_kwargs = self.config['feature_extractor_kwargs']

    # SCHEDULER
    def linear_schedule(self, count):
        frac = (1.0 - (count // (self.config["num_envs"] * self.config['n_train_per_epoch'])))
        return self.config["lr"] * frac

    def save(self,):
        
        # INIT NETWORK
        network = ActorCriticRNN(self.dynamics_model.action_spec().shape[0],
                                 feature_extractor_class=self.feature_extractor ,
                                 feature_extractor_kwargs=self.feature_extractor_kwargs)
        
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

        # init_x = self.extractor.init_x()
        # init_rnn_state_train = ScannedRNN.initialize_carry((self.config["num_envs"], feature_extractor_shape))
        
        # network_params = network.init(random.PRNGKey(self.key), init_rnn_state_train, init_x)
        
        if self.config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(learning_rate=self.linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(self.config["max_grad_norm"]),
                optax.adam(self.config["lr"], eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply,
                                        params=self.params,
                                        tx=tx,
                                        )

        init_rnn_state_eval = ScannedRNN.initialize_carry((self.config["num_envs_eval"], feature_extractor_shape))

        # UPDATE THE SIMULATOR FROM THE LOG
        def _log_step(current_state, unused):

            done = current_state.is_done
            obsv = self.extractor(current_state)
            
            transition = Transition(done,
                                    None,
                                    obsv
            )

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)

            return current_state, transition
        
        # EVALUATION LOOP
        def _evaluate_epoch(train_state):
            
            # EVAL NETWORK
            def _eval_scenario(train_state, scenario):
                
                # Compute the rnn_state on first self.env.config.init_steps from the log trajectory 
                _, log_traj_batch = jax.lax.scan(_log_step, scenario, None, self.env.config.init_steps - 1) 
                rnn_state, _, _ = network.apply(train_state.params, init_rnn_state_eval, (log_traj_batch.obs, log_traj_batch.done))

                current_state = self.env.reset(scenario)

                def extand(x):
                    if isinstance(x, jnp.ndarray):
                        return x[jnp.newaxis, ...]
                    else:
                        return x

                def _eval_step(cary, unused):

                    current_state, rnn_state = cary
                    
                    done = jnp.tile(current_state.is_done, (self.config['num_envs_eval'],))
                    obsv = self.extractor(current_state)
                    
                    # Add a mask here
                    
                    rnn_state, data_action, _ = network.apply(train_state.params,rnn_state,(jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))
                    action = datatypes.Action(data=data_action[0], 
                                              valid=jnp.ones((self.config['num_envs_eval'], 1), dtype='bool'))
                    
                    sdc_obs = datatypes.sdc_observation_from_state(current_state,
                                                roadgraph_top_k=20000)
                    reduced_sdc_obs = jax.tree_map(lambda x : x[0, ...], sdc_obs) # Unbatch
                    
                    img = plot_observation_with_goal(reduced_sdc_obs,
                                                    obj_idx=0,
                                                    goal=obsv['proxy_goal'][0])

                    current_state = self.env.step(current_state, action)
                    
                    metric = self.env.metrics(current_state)

                    return (current_state, rnn_state), (img, metric)

                imgs = []
                metrics = {'log_divergence': [],
                            'overlap': [],
                            'offroad': []}
                
                for _ in range(TRAJ_LENGTH - self.env.config.init_steps):
                    (current_state, rnn_state), (img, metric) = _eval_step((current_state, rnn_state), None)
                    imgs.append(img)
                    for key, value in metric.items():
                        if value.valid:
                            metrics[key].append(value.value)

                return imgs, jax.tree_map(lambda x : jnp.array(x).mean().item(), metrics)

            
            # jit_eval_scenario = jax.jit(_eval_scenario)
            jit_postprocess_fn = jax.jit(self._post_process)

            tt = 0
            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                tt += 1
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC 
                    pass
                else:
                    imgs, metrics = _eval_scenario(train_state, scenario)

                    # Save gif 
                    frames = [Image.fromarray(img) for img in imgs]
                    frames[0].save(os.path.join(self.save_path, f'ex_{tt}_{int(config["IDM"])}.gif'),
                                   save_all=True,
                                   append_images=frames[1:],
                                   duration=100,
                                   loop=0)

                    # Save metric
                    with open(os.path.join(self.save_path, f'ex_{tt}_{int(config["IDM"])}.json'), 'w') as json_file:
                        json.dump(metrics, json_file, indent=4)             
                if tt > self.num_gifs:
                    break

            return None
        
        # Validation
        _evaluate_epoch(train_state)
        
        return None
    

##
# CONFIG
##

parser = argparse.ArgumentParser(description="Agent evaluation")
parser.add_argument('--expe_id', '-expe', type=str, help='Id of the experiment')
parser.add_argument('--epochs', '-e', type=int, help='Number of training epochs')
parser.add_argument('--IDM', '-IDM', type=int, help='Use IDM for simulated agents')
parser.add_argument('--n_evals', '-n', type=int, help='Number of evaluation environments')

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Training config
    load_folder = '/data/draco/cleain/imitation_gap_waymax/logs'
    expe_id = args.expe_id

    with open(os.path.join(load_folder, expe_id, 'args.json'), 'r') as file:
        config = json.load(file)

    config['num_envs_eval'] = args.n_evals

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
    
    config['IDM'] = bool(args.IDM)

    # Env config
    env_config = _config.EnvironmentConfig(
        controlled_object=_config.ObjectType.SDC,
        max_num_objects=config['max_num_obj']
    )

    ##
    # EVALUATION
    ##

    print('Load network parameters')
    n_gifs = 100

    with open(os.path.join(load_folder, expe_id, f'params_{n_epochs}.pkl'), 'rb') as file:
        params = pickle.load(file)

    save_path = os.path.join('/data/draco/cleain/imitation_gap_waymax/animation', expe_id)
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, 'args.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)

    save_gif = save_eval(config, env_config, val_dataset, params, n_gifs, save_path)

    # with jax.disable_jit(): # DEBUG
    save_gif.save()
