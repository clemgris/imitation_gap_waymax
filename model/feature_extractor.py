import flax.linen as nn
import jax.numpy as jnp
from typing import List

from flax.linen.initializers import constant, orthogonal

class FlattenKeyExtractor(nn.Module):
    hidden_layers: int
    keys: List

    @nn.compact
    def __call__(self, obs):
        outputs = []
        for key in self.keys:
            x = obs[key]
            T, B = x.shape[:2]  # Extract dimensions T (time_steps) and B (batch_size) from obs shape
            x = x.reshape((T, B, -1)) # Flatten
            outputs.append(x)
        
        flattened = jnp.concatenate(outputs, axis=-1)

        output = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(flattened)

        return output

