import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import optax
from tqdm import tqdm

from typing import NamedTuple
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents
from waymax.metrics import MetricResult

import sys
sys.path.append('./')

from dataset.config import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from feature_extractor import KeyExtractor
from state_preprocessing import ExtractObs
from rnn_policy import ActorCriticRNN, ScannedRNN
from obs_mask.mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask


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
obs_masks = {
    'ZeroMask': ZeroMask,
    'SpeedGaussianNoise': SpeedGaussianNoise,
    'SpeedUniformNoise': SpeedUniformNoise,
    'SpeedConicObsMask': SpeedConicObsMask
}

class make_eval:
    
    def __init__(self, 
                 config,
                 env_config,
                 val_dataset,
                 params):

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

        # DEFINE OBSERVABILITY MASK
        if 'obs_mask' not in self.config.keys():
            self.config['obs_mask'] = None
            
        if self.config['obs_mask']:
            self.obs_mask = obs_masks[self.config['obs_mask']](**self.config['obs_mask_kwargs'])
        else:
            self.obs_mask = None
            
    # SCHEDULER
    def linear_schedule(self, count):
        n_update_per_epoch = (N_TRAINING * self.config['num_files'] / N_FILES) // self.config["num_envs"]
        n_epoch = jnp.array([count // n_update_per_epoch])
        frac = jnp.where(n_epoch <= 20, 1, 1 / (2**(n_epoch - 20)))
        return self.config["lr"] * frac

    def train(self,):
        
        # INIT NETWORK
        network = ActorCriticRNN(self.dynamics_model.action_spec().shape[0],
                                 feature_extractor_class=self.feature_extractor ,
                                 feature_extractor_kwargs=self.feature_extractor_kwargs)
        
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

        # init_x = self.extractor.init_x()
        # init_rnn_state_train = ScannedRNN.initialize_carry((self.config["num_envs"], feature_extractor_shape))
        
        # network_params = network.init(random.PRNGKey(self.key), init_rnn_state_train, init_x)
        
        if self.config["lr_scheduler"]:
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
        def _log_step(cary, unused):
            
            current_state, rng = cary

            done = current_state.is_done
            # Extract obs in SDC referential
            obs = datatypes.sdc_observation_from_state(current_state,
                                                       roadgraph_top_k=self.config['roadgraph_top_k'])
            
            # Mask
            if self.obs_mask is not None:
                obs = self.obs_mask.mask_obs(current_state, obs, rng)
                
            # Extract the features from the observation
            obsv = self.extractor(current_state, obs)
            
            transition = Transition(done,
                                    None,
                                    obsv
            )

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)
            rng = jax.random.split(random.PRNGKey(rng), num=1)[0, 0]
            
            return (current_state, rng), transition
        
        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):
            
            # EVAL NETWORK
            def _eval_scenario(train_state, scenario, rng):
                
                # Compute the rnn_state on first self.env.config.init_steps from the log trajectory 
                (current_state, rng), log_traj_batch = jax.lax.scan(_log_step, (scenario, rng), None, self.env.config.init_steps - 1) 
                rnn_state, _, _ = network.apply(train_state.params, init_rnn_state_eval, (log_traj_batch.obs, log_traj_batch.done))

                def extand(x):
                    if isinstance(x, jnp.ndarray):
                        return x[jnp.newaxis, ...]
                    else:
                        return x

                def _eval_step(cary, unused):

                    current_state, rnn_state, rng = cary
                    
                    done = current_state.is_done
                    
                    # Extract obs in SDC referential
                    obs = datatypes.sdc_observation_from_state(current_state,
                                                               roadgraph_top_k=self.config['roadgraph_top_k'])
                    # Mask
                    if self.obs_mask is not None:
                        obs = self.obs_mask.mask_obs(current_state, obs, rng)
                        
                    # Extract the features from the observation
                    obsv = self.extractor(current_state, obs)
                    
                    rnn_state, data_action, _ = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))
                    action = datatypes.Action(data=data_action[0], 
                                              valid=jnp.ones((self.config['num_envs_eval'], 1), dtype='bool'))
                    
                    # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                    current_timestep = current_state['timestep']
                    # Squeeze timestep dim
                    current_state = dataclasses.replace(current_state,
                                                        timestep=current_timestep[0])
                    
                    current_state = self.env.step(current_state, action)

                    # Unsqueeze timestep dim
                    current_state = dataclasses.replace(current_state,
                                                        timestep=current_timestep + 1)
                    rng = jax.random.split(random.PRNGKey(rng), num=1)[0, 0]
                    
                    metric = self.env.metrics(current_state)
                    
                    # Save SDC speed
                    _, sdc_idx = jax.lax.top_k(current_state.object_metadata.is_sdc, k=1)
                    sdc_v = jnp.take_along_axis(obs.trajectory.speed, sdc_idx[..., None, None], axis=-2)
                    
                    speed = MetricResult(value=sdc_v, valid=metric['log_divergence'].valid)
                    metric['speed'] = speed

                    return (current_state, rnn_state, rng), metric

                _, scenario_metrics = jax.lax.scan(_eval_step, (current_state, rnn_state, rng), None, TRAJ_LENGTH - self.env.config.init_steps)

                return scenario_metrics
            
            all_metrics = {'log_divergence': [],
                            'overlap': [],
                            'offroad': [],
                            'speed': []}
            
            jit_eval_scenario = jax.jit(_eval_scenario)
            jit_postprocess_fn = jax.jit(self._post_process)

            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC 
                    pass
                else:
                    scenario_metrics = jit_eval_scenario(train_state, scenario, rng)
                    # Reset key
                    rng = jax.random.split(random.PRNGKey(rng), num=1)[0, 0]
                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())
                
            return train_state, all_metrics
        
        metrics = {}
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Validation
            _, val_metric = _evaluate_epoch(train_state, self.key)
            self.key = jax.random.split(random.PRNGKey(self.key), num=1)[0, 0]
            metrics[epoch]['validation'] = val_metric

            val_message = f'Epoch | {epoch} | Val | '
            for key, value in val_metric.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message)
        
        return {"train_state": train_state, "metrics": metrics}