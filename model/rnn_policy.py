import jax
import jax.numpy as jnp
import numpy as np

from typing import Sequence, Dict, Optional, Union

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

import distrax
import functools

class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        breakpoint
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry((ins.shape[0], ins.shape[1])),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=128)(carry, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(input_size):
        # Use a dummy key since the default state init fn is just zeros.
        batch_size, hidden_size = input_size
        return nn.GRUCell(features=128).initialize_carry(jax.random.key(0), (batch_size, hidden_size))
    

class ActorCriticRNN(nn.Module):
    action_dim: int
    config: dict
    feature_extractor_class: nn.Module
    feature_extractor_kwargs: Optional[Union[Dict, None]]

    @nn.compact
    def __call__(self, rnn_state, x):

        obs, dones = x

        # State feature extractor
        state_features = self.feature_extractor_class(**self.feature_extractor_kwargs)(obs)

        # RNN
        rnn_in = (state_features, dones)
        new_rnn_state, x = ScannedRNN()(rnn_state, rnn_in)
        
        # Actor
        x_actor = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x_actor = jax.nn.relu(x_actor)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x_actor)
        # pi = distrax.Categorical(logits=actor_mean) # DISCRET ACTION SPACE
        
        # Critic
        x_critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x_critic = jax.nn.relu(x_critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x_critic)

        # return new_rnn_state, pi, jnp.squeeze(critic, axis=-1) # DISCRETE ACTION SPACE
        return new_rnn_state, actor_mean, jnp.squeeze(critic, axis=-1) # CONTINUOUS ACTION SPACE
    
if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 4,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 5e5,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "CartPole-v1",
        "ANNEAL_LR": True,
        "DEBUG": True,
    }

    rng = jax.random.PRNGKey(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)