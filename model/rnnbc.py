import jax
import jax.numpy as jnp
import optax

from typing import NamedTuple
from flax.training.train_state import TrainState

from waymax import dynamics
from waymax import dataloader
from waymax import datatypes
from waymax import env as _env
from waymax import agents

import sys
sys.path.append('./')

from feature_extractor import XYExtractor
from rnn_policy import ActorCriticRNN, ScannedRNN

class Transition(NamedTuple):
    done: jnp.ndarray
    expert_action: jnp.array
    obs: jnp.ndarray


def make_train(config,
               env_config,
               data_config):
    
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_STEPS"])

    # DATA ITERATOR
    data_iter = dataloader.simulator_state_generator(data_config)

    # DEFINE ENV
    dynamics_model = dynamics.InvertibleBicycleModel()
    dynamics_model = _env.PlanningAgentDynamics(dynamics_model)

    if config['discrete']:
        action_space_dim = dynamics_model.action_spec().shape
        dynamics_model = dynamics.discretizer.DiscreteActionSpaceWrapper(dynamics_model=dynamics_model,
                                                                         bins=config['bins'] * jnp.array(action_space_dim))
    else:
        dynamics_model = dynamics_model

    assert(not config['discrete']) # bug using scan and DiscreteActionWrapper
    
    env = _env.PlanningAgentEnvironment(
        dynamics_model=dynamics_model,
        config=env_config,
        )

    # DEFINE EXPERT AGENT
    expert_agent = agents.create_expert_actor(dynamics_model)

    # SCHEDULER
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train():
        
        # INIT NETWORK
        network = ActorCriticRNN(dynamics_model.action_spec().shape[0],
                                 feature_extractor_class=XYExtractor,
                                 feature_extractor_kwargs={'max_num_obj': config['max_num_obj']},
                                 config=config)
        
        feature_extractor_shape = config['max_num_obj'] * 2 # /!\ Dirty, should be extracted from the network
        
        rng, _rng = jax.random.split(config['KEY'])

        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], config['max_num_obj'], 2)
            ),
            jnp.zeros((1, config["NUM_ENVS"]), dtype=bool),
        )
        init_rnn_state = ScannedRNN.initialize_carry((config["NUM_ENVS"], feature_extractor_shape))
        
        network_params = network.init(_rng, init_rnn_state, init_x)
        
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply,
                                        params=network_params,
                                        tx=tx,
                                        )

        init_rnn_state = ScannedRNN.initialize_carry((config["NUM_ENVS"], feature_extractor_shape))

        # TRAIN LOOP
        def _update_epoch(train_state, unused):

            # UPDATE NETWORK
            def _update_scenario(train_state, unused):

                # GENERATE NEW SCENARIO FROM THE DATASET
                scenario = next(data_iter)

                # INIT ENV
                current_state = env.reset(scenario)
                full_obsv = datatypes.sdc_observation_from_state(current_state,
                                                    roadgraph_top_k=config['roadgraph_top_k'])
                
                obsv = full_obsv.trajectory.xy # /!\ TODO: need to remove the unvalid object
                
                expert_action = expert_agent.select_action(state=current_state,
                                                           actor_state=None,
                                                           params=None,
                                                           rng=None)

                runner_state = (current_state,
                                expert_action,
                                obsv,
                                jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                                )

                # COLLECT TRAJECTORIES FROM scenario
                def _env_step(runner_state, unused):
                    current_state, expert_action, _, _ = runner_state
                    
                    current_state = datatypes.update_state_by_log(current_state, num_steps=1)
                    done = jnp.tile(current_state.is_done, (config['NUM_ENVS'],))

                    full_obsv = datatypes.sdc_observation_from_state(current_state,
                                                                roadgraph_top_k=config['roadgraph_top_k'])
                    
                    obsv = full_obsv.trajectory.xy # /!\ TODO: need to remove the unvalid object

                    expert_action = expert_agent.select_action(state=current_state, params=None, rng=None, actor_state=None)
                    
                    # Add a mask here

                    runner_state = (current_state, expert_action, obsv, done)

                    transition = Transition(done,
                                            expert_action,
                                            obsv
                                            )
                    return runner_state, transition

                # Use jax.lax.scan with the modified _env_step function
                _, traj_batch = jax.lax.scan(f=_env_step, init=runner_state, xs=None, length=config["NUM_STEPS"])
                
                # BACKPROPAGATION ON THE SCENARIO
                def _update(train_state, traj_batch):

                    # def _loss_fn(params, init_rnn_state, traj_batch): # DISCRETE ACTION SPACE
                    #     _, action_dist, _ = network.apply(
                    #         params, init_rnn_state[0], (traj_batch.obs, traj_batch.done)
                    #     )
                    #     log_prob = action_dist.log_prob(traj_batch.expert_action)
                    #     pi_loss = - log_prob

                    #     entropy = action_dist.entropy().mean()

                    #     total_loss = pi_loss

                    #     return total_loss, (entropy)
                    
                    def _loss_fn(params, init_rnn_state, traj_batch): # CONTINUOUS ACTION
                        _, action_values, _ = network.apply(params, init_rnn_state, (traj_batch.obs, traj_batch.done))

                        # Compute the MSE loss
                        expert_action = traj_batch.expert_action.action.data
                        mse_loss = jnp.mean((action_values - expert_action) ** 2)

                        return mse_loss

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=False) # has_aux=True for discrete action
                    total_loss, grads = grad_fn(train_state.params, init_rnn_state, traj_batch)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, total_loss = _update(train_state, traj_batch)

                return train_state, total_loss

            train_state, metric = jax.lax.scan(_update_scenario, train_state, None, config['n_train_per_epoch'])

            return train_state, metric

        rng, _rng = jax.random.split(rng)
        
        train_state, metrics = jax.lax.scan(_update_epoch, train_state, None, config["NUM_EPOCHS"])
        
        return {"train_state": train_state, "metrics": metrics}

    return train