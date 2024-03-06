import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random

import os
import optax
import pickle
from tqdm import tqdm, trange
from typing import NamedTuple
import time

from waymax import agents
from waymax import dataloader
from waymax import datatypes
from waymax import dynamics
from waymax import env as _env

import sys
sys.path.append('./')

from dataset.config import N_TRAINING, N_VALIDATION, TRAJ_LENGTH, N_FILES
from feature_extractor import KeyExtractor
from state_preprocessing import ExtractObs
from rnn_policy import ActorCriticRNN, ScannedRNN
from obs_mask.mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise

class TransitionRL(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    log_prob: jnp.ndarray

class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray

def extand(x):
    if isinstance(x, jnp.ndarray):
        return x[jnp.newaxis, ...]
    else:
        return x


extractors = {
    'ExtractObs': ExtractObs
}
feature_extractors = {
    'KeyExtractor': KeyExtractor
}
obs_masks = {
    'SpeedGaussianNoise': SpeedGaussianNoise,
    'SpeedUniformNoise': SpeedUniformNoise,
    'SpeedConicObsMask': SpeedConicObsMask
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

        # Minibatch
        self.n_minibatch = max([n for n in range(len(self.devices), 0, -1) if self.config['num_envs'] % n == 0])
        self.mini_batch_size = self.config['num_envs'] // self.n_minibatch

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
        self.key = random.PRNGKey(self.config['key'])

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
                                 self.dynamics_model.action_spec().minimum,
                                 self.dynamics_model.action_spec().maximum,
                                 feature_extractor_class=self.feature_extractor ,
                                 feature_extractor_kwargs=self.feature_extractor_kwargs)

        feature_extractor_shape = self.feature_extractor_kwargs['final_hidden_layers']

        init_x = self.extractor.init_x()
        init_rnn_state_train = ScannedRNN.initialize_carry((self.mini_batch_size, feature_extractor_shape))

        network_params = network.init(self.key, init_rnn_state_train, init_x)

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
                                        params=network_params,
                                        tx=tx,
                                        )

        init_rnn_state_eval = ScannedRNN.initialize_carry((self.config["num_envs_eval"], feature_extractor_shape))

        # Jitted functions

        jit_postprocess_fn = jax.jit(self._post_process)

        # UPDATE THE SIMULATOR FROM THE LOG
        def _log_step(cary, unused):

            current_state, rng = cary

            done = current_state.is_done
            # Extract obs in SDC referential
            obs = datatypes.sdc_observation_from_state(current_state,
                                                       roadgraph_top_k=self.config['roadgraph_top_k'])

            # Mask
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                masked_obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)
            else:
                masked_obs = obs

            # Extract the features from the observation
            obsv = self.extractor(current_state, masked_obs)

            transition = Transition(done,
                                    None,
                                    obsv
            )

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)

            return (current_state, rng), transition

        # Divide scenario into minibatches
        def _minibatch(scenario):
            minibatched_scenario = jax.tree_map(lambda x : x.reshape(self.n_minibatch, self.mini_batch_size, *x.shape[1:]), scenario)
            return minibatched_scenario

        # Compute loss, grad on a single minibatch
        def _single_update(train_state, data, rng):

            scenario = jit_postprocess_fn(data)

            # COLLECT TRAJECTORIES FROM scenario
            def _env_step(cary, unused):

                current_state, rng = cary

                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(current_state,
                                                        roadgraph_top_k=self.config['roadgraph_top_k'])

                expert_action = self.expert_agent.select_action(state=current_state,
                                                                params=None,
                                                                rng=None,
                                                                actor_state=None)

                # Mask
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    masked_obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)
                else:
                    masked_obs = obs

                # Extract the features from the observation
                obsv = self.extractor(current_state, masked_obs)

                transition = Transition(done,
                                        expert_action,
                                        obsv)

                # Update the simulator state with the log trajectory
                current_state = datatypes.update_state_by_log(current_state, num_steps=1)

                return (current_state, rng), transition

            # COLLECT TRAJECTORIES FROM scenario
            def _env_step_rl(cary, unused):

                current_state, rnn_state, rng = cary

                done = current_state.is_done

                # Extract obs in SDC referential
                obs = datatypes.sdc_observation_from_state(current_state,
                                                        roadgraph_top_k=self.config['roadgraph_top_k'])
                # Mask
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    masked_obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)
                else:
                    masked_obs = obs

                # Extract the features from the observation
                obsv = self.extractor(current_state, masked_obs)

                rnn_state, action_dist, value = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))

                rng, rng_sample = jax.random.split(rng)
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                log_prob = action_dist.log_prob(action_data)

                action = datatypes.Action(data=action_data,
                                        valid=jnp.ones((self.config['num_envs'], 1), dtype='bool'))

                # Patch bug in waymax (squeeze timestep dimension when using reset --> need squeezed timestep for update)
                current_timestep = current_state['timestep']
                # Squeeze timestep dim
                current_state = dataclasses.replace(current_state,
                                                    timestep=current_timestep[0])

                reward = self.env.reward(current_state, action)

                current_state = self.env.step(current_state, action)

                # Unsqueeze timestep dim
                current_state = dataclasses.replace(current_state,
                                                    timestep=current_timestep + 1)

                transitionRL = TransitionRL(
                    done, action, value, reward, obsv, log_prob
                )

                return (current_state, rnn_state, rng), transitionRL

            # Compute the rnn_state on first self.env.config.init_steps from the log trajectory
            (current_state, rng), log_traj_batch = jax.lax.scan(_log_step, (scenario, rng), None, self.env.config.init_steps - 1)

            # BC
            (_, rng), traj_batch_bc = jax.lax.scan(_env_step, (current_state, rng), None, self.config["num_steps"])

            # RL
            rnn_state, _, _ = network.apply(train_state.params, init_rnn_state_train, (log_traj_batch.obs, log_traj_batch.done))
            (current_state_rl, rnn_state, rng), traj_batch_rl = jax.lax.scan(_env_step_rl, (current_state, rnn_state, rng), None, self.config["num_steps"])

            # Extract last obs and done
            last_done = current_state_rl.is_done

            obs = datatypes.sdc_observation_from_state(current_state_rl,
                                                        roadgraph_top_k=self.config['roadgraph_top_k'])
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                masked_obs = self.obs_mask.mask_obs(current_state_rl, obs, rng_obs)
            else:
                masked_obs = obs

            last_obsv = self.extractor(current_state_rl, masked_obs)

            _, _, last_value = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, last_obsv), last_done[jnp.newaxis, ...]))

            # Compute advantage
            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + self.config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - next_done) * gae
                    return (gae, value, done), gae
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value
            advantages, targets = _calculate_gae(traj_batch_rl, last_value, last_done)

            # BACKPROPAGATION ON THE SCENARIO

            def _loss_fn(params, init_rnn_state, log_traj_batch, traj_batch_bc, traj_batch_rl, advantages, targets, rng):

                def _bc_loss_fn(params, init_rnn_state, log_traj_batch, traj_batch, rng):

                    # Compute the rnn_state from the log on the first steps
                    rnn_state, _, _ = network.apply(params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done))
                    # Compute the action for the rest of the trajectory
                    _, action_dist, _ = network.apply(params, rnn_state, (traj_batch.obs, traj_batch.done))

                    # log_prob = action_dist.log_prob(traj_batch.expert_action.action.data) # LOGPROB
                    # total_loss = - log_prob.mean()

                    action = action_dist.sample(seed=rng) # MSE
                    expert_action = traj_batch.expert_action.action.data
                    total_loss = ((action - expert_action)**2).mean()

                    return total_loss

                def _rl_loss_fn(params, init_rnn_state, log_traj_batch, traj_batch_rl, gae, targets):
                    # RERUN NETWORK

                    # Compute the rnn_state from the log on the first steps
                    rnn_state, _, _ = network.apply(params, init_rnn_state, (log_traj_batch.obs, log_traj_batch.done))
                    # Compute the action for the rest of the trajectory
                    _, pi, value = network.apply(params, rnn_state, (traj_batch_rl.obs, traj_batch_rl.done))

                    log_prob = pi.log_prob(traj_batch_rl.action.data)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch_rl.value + (
                        value - traj_batch_rl.value
                    ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob[:, None, ...] - traj_batch_rl.log_prob)
                    gae = ((gae - gae.mean()) / (gae.std() + 1e-8))[..., None]
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - self.config["CLIP_EPS"],
                            1.0 + self.config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean() # WARNING WHEN CONTINUOUS

                    total_loss = (
                        loss_actor
                        + self.config["VF_COEF"] * value_loss
                        - self.config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                bc_loss = _bc_loss_fn(params, init_rnn_state, log_traj_batch, traj_batch_bc, rng)
                rl_loss, (value_loss, loss_actor, entropy) = _rl_loss_fn(params, init_rnn_state, log_traj_batch, traj_batch_rl, advantages, targets)
                total_loss = (
                    self.config['loss_weight_bc'] * bc_loss
                    + self.config['loss_weight_rl'] * rl_loss
                    )
                return total_loss, (bc_loss, rl_loss, value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True) # has_aux=True for discrete action
            total_loss, grads = grad_fn(train_state.params, init_rnn_state_train, log_traj_batch, traj_batch_bc, traj_batch_rl, advantages, targets, rng)
            return total_loss, grads

        pmap_funct = jax.pmap(lambda x, rng: _single_update(train_state, x, rng))

        # Aggregate losses, grads and update
        def _global_update(train_state, loss, grads):
            mean_grads = jax.tree_map(lambda x: x.mean(0), grads)
            mean_loss = jax.tree_map(lambda x: x.mean(0), loss)

            train_state = train_state.apply_gradients(grads=mean_grads)

            return train_state, mean_loss

        jit_global_update = jax.jit(_global_update)

        # Evaluate
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
                rng, rng_obs = jax.random.split(rng)
                if self.obs_mask is not None:
                    obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

                # Extract the features from the observation
                obsv = self.extractor(current_state, obs)

                rnn_state, action_dist, _ = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))
                rng, rng_sample = jax.random.split(rng)
                action_data = action_dist.sample(seed=rng_sample).squeeze(0)
                action = datatypes.Action(data=action_data,
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

                metric = self.env.metrics(current_state)

                return (current_state, rnn_state, rng), metric

            _, scenario_metrics = jax.lax.scan(_eval_step, (current_state, rnn_state, rng), None, TRAJ_LENGTH - self.env.config.init_steps)

            return scenario_metrics

        jit_eval_scenario = jax.jit(_eval_scenario)

        # TRAIN LOOP
        def _update_epoch(train_state, rng):

            # UPDATE NETWORK
            def _update_scenario(train_state, data, rng):

                minibatched_data = _minibatch(data)
                rng_pmap = jax.random.split(rng, self.n_minibatch)
                loss, grads = pmap_funct(minibatched_data, rng_pmap)

                train_state, mean_loss = jit_global_update(train_state, loss, grads)

                return train_state, mean_loss

            metric = {'loss': [],
                      'bc_loss': [],
                      'rl_loss': []}

            losses = []
            bc_losses = []
            rl_losses = []

            tt = 0
            for data in tqdm(self.train_dataset.as_numpy_iterator(), desc='Training', total=N_TRAINING // self.config['num_envs']):
                tt += 1
            # for _ in trange(1000) # DEBUG:
            #     data = self.data

                rng, rng_train = jax.random.split(rng)
                train_state, (loss, (bc_loss, rl_loss, value_loss, loss_actor, entropy)) = _update_scenario(train_state, data, rng_train)
                losses.append(loss)
                bc_losses.append(bc_loss)
                rl_losses.append(rl_loss)

                if tt > (N_TRAINING * self.config['num_files'] / N_FILES) // self.config['num_envs']:
                    break
            metric['loss'].append(jnp.array(losses).mean())
            metric['bc_loss'].append(jnp.array(bc_losses).mean())
            metric['rl_loss'].append(jnp.array(rl_losses).mean())

            return train_state, metric

        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):

            all_metrics = {'log_divergence': [],
                            'overlap': [],
                            'offroad': []}

            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    scenario_metrics = jit_eval_scenario(train_state, scenario, rng_eval)

                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())

            return train_state, all_metrics

        # LOGS AND CHECKPOINTS
        metrics = {}
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}
            # Training
            self.key, rng_train = jax.random.split(self.key)
            train_state, train_metric = _update_epoch(train_state, rng_train)
            metrics[epoch]['train'] = train_metric

            train_message = f"Epoch | {epoch} | Train | loss | {jnp.array(train_metric['loss']).mean():.4f}"
            train_message += f"| bc_loss | {jnp.array(train_metric['bc_loss']).mean():.4f}"
            train_message += f"| rl_loss | {jnp.array(train_metric['rl_loss']).mean():.4f}"
            print(train_message)

            # Validation
            if (epoch % self.config['freq_eval'] == 0) or (epoch == self.config['num_epochs'] - 1):

                self.key, rng_eval = jax.random.split(self.key)
                _, val_metric = _evaluate_epoch(train_state, rng_eval)
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