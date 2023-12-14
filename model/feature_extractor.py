import flax.linen as nn
import jax.numpy as jnp
from typing import List, Dict

from flax.linen.initializers import constant, orthogonal

class MapExtractor(nn.Module):
    hidden_layers: int

    @nn.compact
    def __call__(self, roadmap_features):
        x = roadmap_features
        x = nn.Conv(self.hidden_layers, kernel_size=(3,))(x)
        nn.relu(x)
        x = nn.max_pool(x, window_shape=(1, 2), strides=(1, 2))

        x = nn.Conv(self.hidden_layers, kernel_size=(3,))(x)
        nn.relu(x)
        x = nn.max_pool(x, window_shape=(1, 2), strides=(1, 2))

        x = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        output = nn.relu(x)

        return output
    
class IdentityExtractor(nn.Module):
    hidden_layers: int = None

    @nn.compact
    def __call__(self, x):
        return x

FEATURES_EXTRACTOR_DICT = {'xy': IdentityExtractor,
                  'proxy_goal': IdentityExtractor,
                  'roadgraph_map': MapExtractor,
                  'traffic_lights': IdentityExtractor}

class KeyExtractor(nn.Module):
    final_hidden_layers: int
    keys: List
    hidden_layers: Dict = None

    @nn.compact
    def __call__(self, obs):
        outputs = []
        for key in self.keys:
            if self.hidden_layers is not None:
                x = FEATURES_EXTRACTOR_DICT[key](self.hidden_layers.get(key, None))(obs[key])
            else:
                x = FEATURES_EXTRACTOR_DICT[key]()(obs[key])
            T, B = x.shape[:2]  # Extract dimensions T (time_steps) and B (batch_size) from obs shape
            x = x.reshape((T, B, -1)) # Flatten
            outputs.append(x)
        
        flattened = jnp.concatenate(outputs, axis=-1)
        output = nn.Dense(self.final_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(flattened)
        return output