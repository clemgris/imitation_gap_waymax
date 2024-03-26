import dataclasses
from flax.training.train_state import TrainState
import functools
import jax
import jax.numpy as jnp
from jax import random
from PIL import Image

import json
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
from model.feature_extractor import KeyExtractor
from model.state_preprocessing import ExtractObs
from model.rnn_policy import ActorCriticRNN, ScannedRNN
from obs_mask.mask import SpeedConicObsMask, SpeedGaussianNoise, SpeedUniformNoise, ZeroMask
from utils.viz import plot_observation_with_goal, plot_observation_with_heading

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
        self.key = random.PRNGKey(self.config['key'])

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
                                 feature_extractor_kwargs=self.feature_extractor_kwargs,
                                 action_minimum=None,
                                 action_maximum=None)

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

        jit_postprocess_fn = jax.jit(self._post_process)

        # UPDATE THE SIMULATOR FROM THE LOG
        def _log_step(cary, rng_extract):

            current_state, rng = cary

            done = current_state.is_done
            # Extract obs in SDC referential
            obs = datatypes.sdc_observation_from_state(current_state,
                                                       roadgraph_top_k=self.config['roadgraph_top_k'])

            # Mask
            rng, rng_obs = jax.random.split(rng)
            if self.obs_mask is not None:
                obs = self.obs_mask.mask_obs(current_state, obs, rng_obs)

            # Extract the features from the observation

            # rng, rng_extract = jax.random.split(rng)
            obsv = self.extractor(current_state, obs, rng_extract)

            transition = Transition(done,
                                    None,
                                    obsv
            )

            # Update the simulator with the log trajectory
            current_state = datatypes.update_state_by_log(current_state, num_steps=1)

            return (current_state, rng), transition

        # Evaluate
        def _eval_scenario(train_state, scenario, rng):

            rng, rng_extract = jax.random.split(rng)
            # Compute the rnn_state on first self.env.config.init_steps from the log trajectory
            (current_state, rng), log_traj_batch = jax.lax.scan(_log_step,
                                                                (scenario, rng),
                                                                rng_extract[None].repeat(self.env.config.init_steps - 1, axis=0),
                                                                self.env.config.init_steps - 1)
            rnn_state, _, _, _, _ ,_ = network.apply(train_state.params, init_rnn_state_eval, (log_traj_batch.obs, log_traj_batch.done))

            def extand(x):
                if isinstance(x, jnp.ndarray):
                    return x[jnp.newaxis, ...]
                else:
                    return x

            def _eval_step(cary, rng_extract):

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
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Sample action and update scenario
                rnn_state, action_dist, _, _, _, _ = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))
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

            def _eval_step_gif(cary, rng_extract):

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
                # rng, rng_extract = jax.random.split(rng)
                obsv = self.extractor(current_state, obs, rng_extract)

                # Generate image

                reduced_sdc_obs = jax.tree_map(lambda x : x[0, ...], obs) # Unbatch
                list_features = self.config['feature_extractor_kwargs']['keys']

                if ('noisy_proxy_goal' in list_features) or ('proxy_goal' in list_features):
                    if 'proxy_goal' in list_features:
                        goal = obsv['proxy_goal'][0]
                    elif 'noisy_proxy_goal' in list_features:
                        goal = obsv['noisy_proxy_goal'][0]

                    img = plot_observation_with_goal(reduced_sdc_obs,
                                                    obj_idx=0,
                                                    goal=goal)

                elif 'heading' in list_features:
                    img = plot_observation_with_heading(reduced_sdc_obs,
                                                        obj_idx=0,
                                                        heading=obsv['heading'].squeeze())
                else:
                    raise ValueError('TODO plot only observation')


                # Sample action and update scenario
                rnn_state, action_dist, _, _, _, _ = network.apply(train_state.params, rnn_state, (jax.tree_map(extand, obsv), done[jnp.newaxis, ...]))
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

                return (current_state, rnn_state, rng), (img, metric)

            imgs = []
            if not self.config['GIF']:

                _, scenario_metrics = jax.lax.scan(_eval_step,
                                                (current_state, rnn_state, rng),
                                                rng_extract[None].repeat(TRAJ_LENGTH - self.env.config.init_steps, axis=0),
                                                TRAJ_LENGTH - self.env.config.init_steps)
            else:
                scenario_metrics = []

                # Loop over timesteps
                for _ in range(TRAJ_LENGTH - self.env.config.init_steps):
                    # Call _eval_step for each timestep
                    (current_state, rnn_state, rng), (img, metric) = _eval_step_gif((current_state, rnn_state, rng), rng_extract)
                    scenario_metrics.append(metric)
                    imgs.append(img)

                # Combine metrics for all timesteps
                scenario_metrics = jax.tree_map(lambda *args: jnp.stack(args), *scenario_metrics)

            return imgs, scenario_metrics

        jit_eval_scenario = jax.jit(_eval_scenario)

        # EVALUATION LOOP
        def _evaluate_epoch(train_state, rng):

            all_metrics = {'log_divergence': [],
                           'max_log_divergence': [],
                           'overlap_rate': [],
                           'overlap': [],
                           'offroad_rate': [],
                           'offroad': []
                           }

            all_metrics_inter = {'log_divergence': [],
                                 'max_log_divergence': [],
                                 'overlap_rate': [],
                                 'overlap': [],
                                 'offroad_rate': [],
                                 'offroad': []
                                 }

            t = 0
            for data in tqdm(self.val_dataset.as_numpy_iterator(), desc='Validation', total=N_VALIDATION // self.config['num_envs_eval'] + 1):
                t += 1
                all_scenario_metrics = {}
                scenario = jit_postprocess_fn(data)
                if not jnp.any(scenario.object_metadata.is_sdc):
                    # Scenario does not contain the SDC
                    pass
                else:
                    rng, rng_eval = jax.random.split(rng)
                    if self.config['GIF']:

                        imgs, scenario_metrics = _eval_scenario(train_state, scenario, rng_eval)

                        frames = [Image.fromarray(img) for img in imgs]
                        frames[0].save(os.path.join('/data/tucana/cleain/imitation_gap_waymax/animation/', self.config['log_folder'][5:], f'ex_{t}.gif'),
                                    save_all=True,
                                    append_images=frames[1:],
                                    duration=100,
                                    loop=0)
                    else:
                        _, scenario_metrics = jit_eval_scenario(train_state, scenario, rng_eval)

                    for key, value in scenario_metrics.items():
                        if jnp.any(value.valid):
                            all_metrics[key].append(value.value[value.valid].mean())
                            all_scenario_metrics[key] = value.value[value.valid].mean()
                        if jnp.any(scenario.object_metadata.objects_of_interest):
                            has_inter = jnp.any(scenario.object_metadata.objects_of_interest, axis=1)[None].repeat(80, axis=0)
                            all_metrics_inter[key].append(value.value[value.valid & has_inter].mean())

                    key = 'max_log_divergence'
                    value = scenario_metrics['log_divergence']
                    if jnp.any(value.valid):
                        all_metrics[key].append(value.value[value.valid].max())
                        all_scenario_metrics[key] = value.value[value.valid].max()
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(scenario.object_metadata.objects_of_interest, axis=1)[None].repeat(80, axis=0)
                        all_metrics_inter[key].append(value.value[value.valid & has_inter].max())

                    key = 'overlap_rate'
                    value = scenario_metrics['overlap']
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value[value.valid]))
                        all_scenario_metrics[key] = jnp.any(value.value[value.valid])
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(scenario.object_metadata.objects_of_interest, axis=1)[None].repeat(80, axis=0)
                        all_metrics_inter[key].append(jnp.any(value.value[value.valid & has_inter]))

                    key = 'offroad_rate'
                    value = scenario_metrics['offroad']
                    if jnp.any(value.valid):
                        all_metrics[key].append(jnp.any(value.value[value.valid]))
                        all_scenario_metrics[key] = jnp.any(value.value[value.valid])
                    if jnp.any(scenario.object_metadata.objects_of_interest):
                        has_inter = jnp.any(scenario.object_metadata.objects_of_interest, axis=1)[None].repeat(80, axis=0)
                        all_metrics_inter[key].append(jnp.any(value.value[value.valid & has_inter]))

                    if self.config['GIF']:
                        folder = os.path.join(os.path.join(f'/data/tucana/cleain/imitation_gap_waymax/animation/', self.config['log_folder'][5:], f'ex_{t}.json'))
                        with open(folder, 'w') as json_file:
                            json.dump(jax.tree_map(lambda x : x.item(), all_scenario_metrics), json_file, indent=4)

            return train_state, (all_metrics, all_metrics_inter)

        metrics = {}
        rng = self.key
        for epoch in range(self.config["num_epochs"]):
            metrics[epoch] = {}

            # Validation
            rng, rng_eval = jax.random.split(rng)
            _, (val_metric, val_metric_inter) = _evaluate_epoch(train_state, rng_eval)
            metrics[epoch]['validation'] = val_metric

            val_message = f'Epoch | {epoch} | Val | '
            for key, value in val_metric.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message)

            val_message = f'Epoch | {epoch} | Val | '
            for key, value in val_metric_inter.items():
                val_message += f" {key} | {jnp.array(value).mean():.4f} | "

            print(val_message)

        return {"train_state": train_state, "metrics": metrics}