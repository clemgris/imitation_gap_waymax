import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Any, List, Dict

import sys
sys.path.append('../')

from dataset.config import NUM_POLYLINE_TYPES

from flax.linen.initializers import constant, orthogonal

class MapEncoder(nn.Module):
    hidden_layers: int

    @nn.compact
    def __call__(self, roadmap_features):
        x = roadmap_features

        x = nn.Conv(self.hidden_layers, kernel_size=(3,))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(1, 2), strides=(1, 2))

        x = nn.Conv(self.hidden_layers, kernel_size=(3,))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(1, 2), strides=(1, 2))

        x = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        output = nn.relu(x)

        return output

class PolylineEncoder(nn.Module):
    hidden_layers: int
    poly_hidden_layers: int=32
    out_channels: int=None

    @nn.compact
    def __call__(self, points) -> Any:

        def single_call(points):

            B, N, _ = points.shape # Batch_size, number of points, num_features

            points = points.reshape((B * N, -1)) # (B * N, C)
            valid = points[:, -1].reshape((B * N, -1)) # (B * N, 1)
            types = jnp.argmax(points[:, 4:-1].reshape((B * N, -1)), axis=1, keepdims=True) # (B * N, 1)

            # Points features
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points) # (B * N, H)
            points_features = jax.nn.relu(points_features)
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points_features) # (B * N, H)
            points_features = jax.nn.relu(points_features)
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points_features) # (B * N, H)
            points_features = jax.nn.relu(points_features)

            H = self.poly_hidden_layers
            points_features = jnp.where(valid.repeat(H, axis=-1),
                                        points_features,
                                        jnp.zeros((B * N, H))
                                        ) # (B * N, H)

            # Polylines features
            polylines_features = jnp.zeros((B, N, H)) # (B, N, H)
            for type in range(NUM_POLYLINE_TYPES):
                polyline_point_features = jnp.where(types == type,
                                                    points_features,
                                                    jnp.zeros((B * N, H))) # (B * N, H)

                pooled_features = polyline_point_features.reshape((B, N, -1)).max(axis=-2, keepdims=True) # (B, 1, H)

                polylines_features = jnp.where(types.reshape((B, N, -1)).repeat(H, axis=-1) == type,
                                            pooled_features.repeat(N, -2),
                                            polylines_features) # (B, N, H)


            points_features = jnp.concatenate((points_features.reshape((B, N, -1)), polylines_features), axis=-1) # Add global to local features
            points_features = points_features.reshape((B * N, -1)) # (B * N, 2 * H)


            # Augmented points features
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points_features) # (B * N, H)
            points_features = jax.nn.relu(points_features)
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points_features) # (B * N, H)
            points_features = jax.nn.relu(points_features)
            points_features = nn.Dense(self.poly_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(points_features) # (B * N, H)
            points_features = jax.nn.relu(points_features)

            H = self.poly_hidden_layers
            points_features = jnp.where(valid.repeat(H, axis=-1),
                                        points_features,
                                        jnp.zeros((B * N, H))
                                        ) # (B * N, H)

            # Polylines features
            all_pooled_features = []
            for type in range(NUM_POLYLINE_TYPES):
                polyline_point_features = jnp.where(types == type,
                                                    points_features,
                                                    jnp.zeros((B * N, H))) # (B * N, H)

                pooled_features = polyline_point_features.reshape((B, N, -1)).max(axis=-2, keepdims=True) # (B, 1, H)
                all_pooled_features.append(pooled_features)

            return jnp.concatenate(all_pooled_features, axis=1) # (B, NUM_POLYLINE_TYPES, H)

        output = jax.vmap(single_call)(points) # (T, B, NUM_POLYLINE_TYPES, H)
        T, B = output.shape[:2]
        output = output.reshape((T, B, -1)) # (T, B, NUM_POLYLINE_TYPES * H)
        output = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(output)
        output = jax.nn.relu(output)

        return output

class IdentityEncoder(nn.Module):
    hidden_layers: int = None

    @nn.compact
    def __call__(self, x):
        return x

class MlpEncoder(nn.Module):
    hidden_layers: int = None

    @nn.compact
    def __call__(self, x):
        T, B = x.shape[:2]  # Extract dimensions T (time_steps) and B (batch_size) from obs shape
        x = x.reshape((T, B, -1)) # Flatten
        output = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        output = jax.nn.relu(output)
        output = nn.Dense(self.hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(output)
        output = jax.nn.relu(output)
        return output

FEATURES_EXTRACTOR_DICT = {'xy': MlpEncoder,
                           'xyyaw': MlpEncoder,
                           'xyyawv': MlpEncoder,
                           'sdc_speed': MlpEncoder,
                           'proxy_goal': MlpEncoder,
                           'noisy_proxy_goal': MlpEncoder,
                           'heading': MlpEncoder,
                           'roadgraph_map': PolylineEncoder,
                           'traffic_lights': MlpEncoder}

class KeyExtractor(nn.Module):
    final_hidden_layers: int
    keys: List
    kwargs: Dict = None
    hidden_layers: Dict = None

    @nn.compact
    def __call__(self, obs):
        outputs = []
        for key in self.keys:
            if self.hidden_layers is not None:
                x = FEATURES_EXTRACTOR_DICT[key](self.hidden_layers.get(key, None))(obs[key])
            else:
                x = FEATURES_EXTRACTOR_DICT[key]()(obs[key])
            x = nn.LayerNorm()(x) # Layer norm
            outputs.append(x)

        flattened = jnp.concatenate(outputs, axis=-1)

        output = nn.Dense(self.final_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(flattened)
        output = jax.nn.relu(output)
        output = nn.Dense(self.final_hidden_layers, kernel_init=orthogonal(2), bias_init=constant(0.0))(output)
        output = jax.nn.relu(output)
        return output