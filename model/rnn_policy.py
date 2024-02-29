import distrax
import jax
import jax.numpy as jnp
import numpy as np

from typing import Dict, Optional, Union
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
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
    action_minimum: jnp.ndarray
    action_maximum: jnp.ndarray
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
        actor_std = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x_actor)
        actor_std = nn.softplus(actor_std)
        
        # pi = distrax.Categorical(logits=actor_mean) # DISCRET ACTION SPACE
        pi = tfd.TruncatedNormal(loc=actor_mean, 
                                scale=actor_std,
                                low=self.action_minimum,
                                high=self.action_maximum)
        
        # Critic
        x_critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        x_critic = jax.nn.relu(x_critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x_critic)

        # return new_rnn_state, pi, jnp.squeeze(critic, axis=-1) # DISCRETE ACTION SPACE
        return new_rnn_state, pi, jnp.squeeze(critic, axis=-1) # CONTINUOUS ACTION SPACE