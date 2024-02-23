from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any, Dict

from model.config import UNVALID_MASK_VALUE
from model.model_utils import combine_two_object_pose_2d, radius_point_extra
from waymax import datatypes
from waymax.datatypes import transform_trajectory

from dataset.config import HEADING_RADIUS

from utils.observation import last_sdc_observation_for_current_sdc_from_state

def extract_xy(state, obs):
    """Extract the xy positions of the object of the scene.
    Mask the unvalid objects with a default value (UNVALID_MASK_VALUE).

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local 
        referential.

    Returns:
        Masked xy positions (trajectory of size 1) of the 
        objects in the scene.
    """
    xy = obs.trajectory.xy

    valid = obs.trajectory.valid[..., None]
    masked_traj = jnp.where(valid, 
                            xy, 
                            UNVALID_MASK_VALUE * jnp.ones_like(xy))
    
    return masked_traj

def extract_goal(state, obs):
    """Generates the proxy goal as the last 
    xy positin of the SDC in the log trajectory.

    Args:
        state: The current simulator state.
        obs: The current observation of the SDC in its local referential (unused).

    Returns:
        Proxy goal of shape (..., 2). Note that the proxy 
        goal coordinates are in the referential of the current
        SDC position.
    """
    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

    last_sdc_obs = last_sdc_observation_for_current_sdc_from_state(state) # Last obs of the log in the current SDC pos referential
    last_sdc_xy = jnp.take_along_axis(last_sdc_obs.trajectory.xy[..., 0, :], sdc_idx[..., None, None], axis=-2)
    
    mask = jnp.any(state.object_metadata.is_sdc)[..., None, None] # Mask if scenario with no SDC
    proxy_goal = last_sdc_xy * mask

    return proxy_goal

def extract_heading(state, obs, radius=HEADING_RADIUS):
    """Generates the heading for the SDC to move towards 
    the log position radius meters away from its 
    current position.

    Args:
        state: Has shape (...,).
        radius: the considered distance in meters.

    Returns:
        Heading with shape (..., 2). Note that the heading is in
        the coordinate system of the current SDC position.
    """
    def proxy_heading(state: datatypes.simulator_state.SimulatorState,
                      radius: float
                      ) -> jnp.ndarray:
        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1) 

        obj_xy = state.current_sim_trajectory.xy[..., 0, :]
        obj_yaw = state.current_sim_trajectory.yaw[..., 0]
        obj_valid = state.current_sim_trajectory.valid[..., 0]

        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)

        current_sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., None], axis=-2)
        current_sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
        current_sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)

        current_sdc_pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
            xy=current_sdc_xy, yaw=current_sdc_yaw, valid=current_sdc_valid
        )

        traj = state.log_trajectory

        pose2d_shape = traj.shape
        # Global coordinates pose2d
        pose2d = datatypes.ObjectPose2D.from_center_and_yaw(
            xy=jnp.zeros(shape=pose2d_shape + (2,)),
            yaw=jnp.zeros(shape=pose2d_shape),
            valid=jnp.ones(shape=pose2d_shape, dtype=jnp.bool_),
        )
        pose = combine_two_object_pose_2d(
            src_pose=pose2d, dst_pose=current_sdc_pose2d
        )

        # Project log trajectory in the ref of the current SDC
        transf_traj = transform_trajectory(traj, pose)

        sdc_transf_traj_xy = jnp.take_along_axis(transf_traj.xy, sdc_idx[..., None, None], axis=0)
        sdc_transf_traj_yaw = jnp.take_along_axis(transf_traj.yaw, sdc_idx[..., None], axis=0)
        sdc_transf_traj_yaw = jnp.take_along_axis(transf_traj.yaw, sdc_idx[..., None], axis=0)

        # Compute dist to the current SDC pos
        dist_matrix = jnp.linalg.norm(sdc_transf_traj_xy, axis=-1)
        dist_matrix = jnp.where((jnp.arange(91) < state.timestep)[None, ...],
                            jnp.zeros_like(dist_matrix), 
                            dist_matrix)

        # Intersection btw the circle and log trajectory
        _, idx_radius_point = jax.lax.top_k(dist_matrix > radius, k=1)
        radius_point = jnp.take_along_axis(sdc_transf_traj_xy, idx_radius_point[..., None], axis=-2)
        
        inter_heading =  radius_point / jnp.linalg.norm(radius_point, axis=-1)

        # Intersection btw the circle and extrapolation of the log trajectory
        last_sdc_xy = sdc_transf_traj_xy[:, -1, :]
        last_sdc_yaw = sdc_transf_traj_yaw[:, -1]
        last_sdc_heading = jnp.stack((jnp.cos(last_sdc_yaw), jnp.sin(last_sdc_yaw)), axis=-1)

        extra_heading = radius_point_extra(last_sdc_xy, last_sdc_heading, jnp.zeros_like(last_sdc_xy), radius)

        current_sdc_heading = jnp.where(idx_radius_point == 0, extra_heading, inter_heading)
        
        # assert(jnp.any(current_sdc_heading != jnp.zeros((2,)), axis=-1))
        
        current_sdc_heading = current_sdc_heading / jnp.linalg.norm(current_sdc_heading)

        return current_sdc_heading
    state
    return jax.vmap(proxy_heading, (0, None))(state, radius)

def extract_roadgraph(state, obs):
    """Extract the features (xy, dir_xy, type) of the roadgraph 
    points. Mask the unvalid objects with a default value (UNVALID_MASK_VALUE).

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local 
        referential.

    Returns:
        Masked roadgraph points features (trajectory of size 1).
    """
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
    """Extract the features (xy positions, type) of the traffic 
    lights present in the scene. Mask the unvalid objects with
    a default value (UNVALID_MASK_VALUE).

    Args:
        state: The current simulator state (unused).
        obs: The current observation of the SDC in its local 
        referential.

    Returns:
        Masked traffic lights features (trajectory of size 1).
    """
    traffic_lights = obs.traffic_lights.xy
    valid = obs.traffic_lights.valid[..., None]
    masked_traffic_lights = jnp.where(valid, traffic_lights, UNVALID_MASK_VALUE * jnp.ones_like(traffic_lights))

    traffic_lights_type = obs.traffic_lights.state
    traffic_lights_type_one_hot = jax.nn.one_hot(traffic_lights_type, 9)

    traffic_lights_features = jnp.concatenate((masked_traffic_lights, traffic_lights_type_one_hot), axis=-1)
    
    return traffic_lights_features

EXTRACTOR_DICT = {'xy': extract_xy,
                  'proxy_goal': extract_goal,
                  'heading': extract_heading,
                  'roadgraph_map': extract_roadgraph,
                  'traffic_lights': extract_trafficlights}

def init_dict(config):
    return {'xy': jnp.zeros((1, config["num_envs"], config['max_num_obj'], 2)),
            'proxy_goal': jnp.zeros((1, config["num_envs"], 2)),
            'heading': jnp.zeros((1, config["num_envs"], 2)),
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

    def __call__(self, state, obs):
        obs_features = {}
        
        for key in self.config['feature_extractor_kwargs']['keys']:
            obs_feature = EXTRACTOR_DICT[key](state, obs)
            B = obs_feature.shape[0]
            obs_feature = jnp.squeeze(obs_feature)
            if B == 1:
                obs_feature = obs_feature[jnp.newaxis, ...]
            obs_features[key] = obs_feature
        return obs_features
    
    def init_x(self,):
        init_obs = {}
        all_init_dict = init_dict(self.config)
        for key in self.config['feature_extractor_kwargs']['keys']:
            init_obs[key] = all_init_dict[key]
        return  (init_obs,
                 jnp.zeros((1, self.config["num_envs"]), dtype=bool),
                 )