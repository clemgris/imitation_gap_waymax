from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any, Dict

from model.config import UNVALID_MASK_VALUE
from waymax import datatypes

from utils.observation import last_sdc_observation_for_current_sdc_from_state


def extract_xy(state, obs):
    traj = obs.trajectory.xy

    valid = obs.trajectory.valid[..., None]
    masked_traj = jnp.where(valid, traj, UNVALID_MASK_VALUE * jnp.ones_like(traj))
    
    return masked_traj

def extract_goal(state, obs):
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

    last_sdc_obs = last_sdc_observation_for_current_sdc_from_state(state) # Last obs of the log in the current SDC pos referential
    last_sdc_xy = jnp.take_along_axis(last_sdc_obs.trajectory.xy[..., 0, :], sdc_idx[..., None, None], axis=-2)
    
    mask = jnp.any(state.object_metadata.is_sdc)[..., None, None] # Mask if scenario with no SDC
    proxy_goal = last_sdc_xy * mask

    return proxy_goal

def extract_roadgraph(state, obs):
    valid_roadmap_point = obs.roadgraph_static_points.valid[..., None]

    roadmap_point = obs.roadgraph_static_points.xy
    masked_roadmap_point = jnp.where(valid_roadmap_point, roadmap_point, UNVALID_MASK_VALUE * jnp.ones_like(roadmap_point))

    roadmap_dir = obs.roadgraph_static_points.dir_xy
    masked_roadmap_dir = jnp.where(valid_roadmap_point, roadmap_dir, UNVALID_MASK_VALUE * jnp.ones_like(roadmap_dir))

    roadmap_type = obs.roadgraph_static_points.types
    roadmap_type_one_hot = jax.nn.one_hot(roadmap_type, 20)

    roadmap_point_features = jnp.concatenate((masked_roadmap_point, masked_roadmap_dir, roadmap_type_one_hot), axis=-1)
    
    return roadmap_point_features

def extract_trafficlights(state, obs):
    traffic_lights = obs.traffic_lights.xy
    valid = obs.traffic_lights.valid[..., None]
    masked_traffic_lights = jnp.where(valid, traffic_lights, UNVALID_MASK_VALUE * jnp.ones_like(traffic_lights))

    traffic_lights_type = obs.traffic_lights.state
    traffic_lights_type_one_hot = jax.nn.one_hot(traffic_lights_type, 9)

    traffic_lights_features = jnp.concatenate((masked_traffic_lights, traffic_lights_type_one_hot), axis=-1)
    
    return traffic_lights_features

EXTRACTOR_DICT = {'xy': extract_xy,
                  'proxy_goal': extract_goal,
                  'roadgraph_map': extract_roadgraph,
                  'traffic_lights': extract_trafficlights}

def init_dict(config):
    return {'xy': jnp.zeros((1, config["num_envs"], config['max_num_obj'], 2)),
            'proxy_goal': jnp.zeros((1, config["num_envs"], 2)),
            'roadgraph_map': jnp.zeros((1, config["num_envs"], config['roadgraph_top_k'], 24)),
            'traffic_lights': jnp.zeros((1, config["num_envs"], 16, 11))
            }

class Extractor(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
        
    @abstractmethod
    def init_x(self):
        pass

@dataclass
class ExtractObs(Extractor):
    config: Dict

    def __call__(self, state):
        obs_features = {}
        obs = datatypes.sdc_observation_from_state(state,
                                                   roadgraph_top_k=self.config['roadgraph_top_k'])
        for key in self.config['feature_extractor_kwargs']['keys']:
            obs_features[key] = jnp.squeeze(EXTRACTOR_DICT[key](state, obs))
        return obs_features
    
    def init_x(self,):
        init_obs = {}
        all_init_dict = init_dict(self.config)
        for key in self.config['feature_extractor_kwargs']['keys']:
            init_obs[key] = all_init_dict[key]
        return  (init_obs,
                 jnp.zeros((1, self.config["num_envs"]), dtype=bool),
                 )