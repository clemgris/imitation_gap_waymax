import jax
import jax.numpy as jnp
import numpy as np
import dataclasses

import flax.linen as nn


class XYExtractor(nn.Module):
    def __init__(self, max_num_obj: int):
        super().__init__()
        self.max_num_obj = max_num_obj

    def setup(self):
        # Define the setup method to set the feature shape
        self._feature_shape = self.max_num_obj * 2

    def __call__(self, obs):
        T, B = obs.shape[:2]  # Extract dimensions T (time_steps) and B (batch_size) from obs shape
        return obs.reshape((T, B, -1)) # Flatten
