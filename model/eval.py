from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import os
import optax
import pickle
from tqdm import tqdm

from typing import NamedTuple
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents

import sys
sys.path.append('./')

from dataset.config import  N_VALIDATION, TRAJ_LENGTH
from feature_extractor import FlattenKeyExtractor
from state_preprocessing import ExtractXY, ExtractXYGoal
from rnn_policy import ActorCriticRNN, ScannedRNN


class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

extractors = {
    'ExtractXY': ExtractXY,
    'ExtractXYGoal': ExtractXYGoal,
}
feature_extractors = {
    'FlattenKeyExtractor': FlattenKeyExtractor
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
        self.wrapped_dynamics_model = dynamics.InvertibleBicycleModel()
        self.dynamics_model = _env.PlanningAgentDynamics(self.wrapped_dynamics_model)

        if config['discrete']:
            action_space_dim = self.dynamics_model.action_spec().shape
            self.dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=self.dynamics_model,
                                                                                  bins=config['bins'] * jnp.array(action_space_dim))
        else:
            self.dynamics_model = self.dynamics_model

        assert(not config['discrete']) # /!\ BUG using scan and DiscreteActionWrapper
        
        self.env = _env.PlanningAgentEnvironment(
            dynamics_model=self.wrapped_dynamics_model,
            config=env_config,
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

    def train(self,):
        
        # INIT NETWORK
        network = ActorCriticRNN(self.dynamics_model.action_spec().shape[0],
                                 feature_extractor_class=self.feature_extractor ,
                                 feature_extractor_kwargs=self.feature_extractor_kwargs,
                                 config=self.config)
        
        feature_extractor_shape = self.feature_extractor_kwargs['hidden_layers']

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

                    current_state = self.env.step(current_state, action)
                    
                    metric = self.env.metrics(current_state)

                    return (current_state, rnn_state), metric

                _, scenario_metrics = jax.lax.scan(_eval_step, (current_state, rnn_state), None, TRAJ_LENGTH - self.env.config.init_steps)

                return scenario_metrics
            
            all_metrics = {'log_divergence': [],
                            'overlap': [],
                            'offroad': []}
            
            jit_eval_scenario = jax.jit(_eval_scenario)
            jit_postprocess_fn = jax.jit(self._post_process)

            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC 
                    pass
                else:
                    scenario_metrics = jit_eval_scenario(train_state, scenario)
                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())

            return train_state, all_metrics    
        
        metrics = {}
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Validation
            _, val_metric = _evaluate_epoch(train_state)
            metrics[epoch]['validation'] = val_metric

            val_message = f'Epoch | {epoch} | Val | '
            for key, value in val_metric.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message)
        
        return {"train_state": train_state, "metrics": metrics}