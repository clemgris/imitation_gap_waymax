from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import time

import os
import optax
import pickle
from tqdm import tqdm, trange

from typing import NamedTuple
from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents

import sys
sys.path.append('./')

from dataset.config import N_TRAINING, N_VALIDATION, TRAJ_LENGTH
from feature_extractor import KeyExtractor
from state_preprocessing import ExtractObs
from rnn_policy import ActorCriticRNN, ScannedRNN


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

class make_train:
    
    def __init__(self, 
                 config,
                 env_config,
                 train_dataset,
                 val_dataset,
                 data # DEBUG
                 ):
        
        self.data = data # DEBUG

        self.config = config
        self.env_config = env_config

        # Device
        self.devices = jax.devices()
        print(f'Available devices: {self.devices}')

        # Postprocessing function
        self._post_process = functools.partial(
            dataloader.womd_factories.simulator_state_from_womd_dict,
            include_sdc_paths=config['include_sdc_paths'],
              )
        
        # TRAINING DATASET
        self.train_dataset = train_dataset

        # VALIDATION DATASET
        self.val_dataset = val_dataset
        
        # Random key
        self.key = self.config['key']

        # DEFINE ENV
        if 'dynamics' not in self.config.keys():
            self.config['dynamics'] = 'bicycle'
        
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
        frac = (1.0 - (count // (self.config["num_envs"] * N_TRAINING))) # TODO: add the number of epochs
        return self.config["lr"] * frac

    def train(self,):
        
        # INIT NETWORK
        network = ActorCriticRNN(self.dynamics_model.action_spec().shape[0],
                                 feature_extractor_class=self.feature_extractor ,
                                 feature_extractor_kwargs=self.feature_extractor_kwargs)
        
        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

        init_x = self.extractor.init_x()
        init_rnn_state_train = ScannedRNN.initialize_carry((self.config["num_envs"], feature_extractor_shape))
        
        network_params = network.init(random.PRNGKey(self.key), init_rnn_state_train, init_x)
        
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
                                        params=network_params,
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

        # TRAIN LOOP
        def _update_epoch(train_state):

            # UPDATE NETWORK
            def _update_scenario(train_state, scenario):

                # INIT ENV
                current_state = self.env.reset(scenario)               

                # COLLECT TRAJECTORIES FROM scenario
                def _env_step(current_state, unused):
                    
                    done = jnp.tile(current_state.is_done, (self.config['num_envs'],))
                    obsv = self.extractor(current_state)

                    expert_action = self.expert_agent.select_action(state=current_state, params=None, rng=None, actor_state=None)
                    
                    # Add a mask here

                    transition = Transition(done,
                                            expert_action,
                                            obsv
                                            )
                    
                    # Update the simulator state with the log trajectory
                    current_state = datatypes.update_state_by_log(current_state, num_steps=1)

                    return current_state, transition
                
                # Compute the rnn_state on first self.env.config.init_steps from the log trajectory 
                _, log_traj_batch = jax.lax.scan(_log_step, scenario, None, self.env.config.init_steps - 1) 

                # Use jax.lax.scan with the modified _env_step function
                _, traj_batch = jax.lax.scan(_env_step, current_state, None, self.config["num_steps"])

                # BACKPROPAGATION ON THE SCENARIO
                def _update(train_state, log_traj_batch, traj_batch):

                    # def _loss_fn(params, init_rnn_state, traj_batch): # DISCRETE ACTION SPACE
                    #     _, action_dist, _ = network.apply(
                    #         params, init_rnn_state[0], (traj_batch.obs, traj_batch.done)
                    #     )
                    #     log_prob = action_dist.log_prob(traj_batch.expert_action)
                    #     pi_loss = - log_prob

                    #     entropy = action_dist.entropy().mean()

                    #     total_loss = pi_loss

                    #     return total_loss, (entropy)
                    
                    def _loss_fn(params, init_rnn_state, log_traj_batch, traj_batch): # CONTINUOUS ACTION SPACE
                        
                        # Compute the rnn_state from the log on the first steps
                        rnn_state, _, _ = network.apply(params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done))
                        # Compute the action for the rest of the trajectory
                        _, action_values, _ = network.apply(params, rnn_state, (traj_batch.obs, traj_batch.done))

                        # Compute the MSE loss
                        expert_action = traj_batch.expert_action.action.data
                        mse_loss = jnp.mean((action_values - expert_action) ** 2)

                        return mse_loss

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=False) # has_aux=True for discrete action
                    total_loss, grads = grad_fn(train_state.params, init_rnn_state_train, log_traj_batch, traj_batch)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, total_loss = _update(train_state, log_traj_batch, traj_batch)

                return train_state, total_loss
                        
            metric = {'loss': []}
            
            losses = []
            jit_update_scenario = jax.jit(_update_scenario)
            jit_postprocess_fn = jax.jit(self._post_process)

            for data in tqdm(self.train_dataset.as_numpy_iterator(), desc='Training', total=N_TRAINING // self.config['num_envs']): 
            # for _ in trange(1000) # DEBUG:
            #     data = self.data
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC 
                    pass
                else:
                    train_state, loss = jit_update_scenario(train_state, scenario)
                    losses.append(loss)
            metric['loss'].append(jnp.array(losses).mean())
            
            # SUFFLE TRAINING DATA ITERATOR
            self.key = jax.random.split(random.PRNGKey(self.key), num=1)[0, 0]
            self.train_dataset.shuffle(self.config['shuffle_buffer_size'], self.key)

            return train_state, metric
        
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
            # Training
            train_state, train_metric = _update_epoch(train_state)
            metrics[epoch]['train'] = train_metric

            train_message = f"Epoch | {epoch} | Train | loss | {jnp.array(train_metric['loss']).mean():.4f}"
            print(train_message)

            # Validation
            if (epoch % self.config['freq_eval'] == 0) or (epoch == self.config['num_epochs'] - 1):
                _, val_metric = _evaluate_epoch(train_state)
                metrics[epoch]['validation'] = val_metric

                val_message = f'Epoch | {epoch} | Val | '
                for key, value in val_metric.items():
                    val_message += f" {key} | {jnp.array(value).mean():.4f} | "

                print(val_message)

            if (epoch % self.config['freq_save'] == 0) or (epoch == self.config['num_epochs'] - 1):
                past_log_metric = os.path.join(self.config['log_folder'], f'training_metrics_{epoch - self.config["freq_save"]}.pkl')
                past_log_params = os.path.join(self.config['log_folder'], f'params_{epoch - self.config["freq_save"]}.pkl')
                
                if os.path.exists(past_log_metric):
                    os.remove(past_log_metric)

                if os.path.exists(past_log_params):
                    os.remove(past_log_params)

                # Checkpoint
                with open(os.path.join(self.config['log_folder'], f'training_metrics_{epoch}.pkl'), "wb") as pkl_file:
                    pickle.dump(metrics, pkl_file)

                # Save model weights
                with open(os.path.join(self.config['log_folder'], f'params_{epoch}.pkl'), 'wb') as f:
                    pickle.dump(train_state.params, f)
        
        return {"train_state": train_state, "metrics": metrics}