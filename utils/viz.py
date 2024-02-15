from typing import Any, Optional, Callable

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

from waymax import config as waymax_config
from waymax import datatypes
from waymax.visualization import utils

from waymax.visualization import plot_trajectory, plot_roadgraph_points, plot_traffic_light_signals_as_points

def plot_observation_with_mask(
    obs: datatypes.Observation,
    obj_idx: int,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC,
    mask_function: Optional[Callable[[matplotlib.axes.Axes], None]] = None
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  # Plot the mask
  mask_function(ax)

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))
  
  return utils.img_from_fig(fig)


def plot_observation_with_goal(
    obs: datatypes.Observation,
    obj_idx: int,
    goal: tuple,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  ax.scatter(goal[0], goal[1], marker='X', c='blue')

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))
  
  return utils.img_from_fig(fig)


def plot_observation_with_heading(
    obs: datatypes.Observation,
    obj_idx: int,
    heading: jnp.ndarray,
    viz_config: Optional[dict[str, Any]] = None,
    batch_idx: int = -1,
    highlight_obj: waymax_config.ObjectType = waymax_config.ObjectType.SDC
) -> np.ndarray:
  """Plots np array image for an Observation.

  Args:
    obs: An Observation instance, with shape (..., obs_A), where obs_A
      represents the number of objects that have observation view over things
      including other objects, roadgraph, and traffic lights.
    obj_idx: The object index in obs_A.
    viz_config: Dict for optional config.
    batch_idx: Optional batch index.
    highlight_obj: Represents the type of objects that will be highlighted with
      `color.COLOR_DICT['controlled']` color.

  Returns:
    np image.
  """
  if batch_idx > -1:
    if len(obs.shape) != 2:
      raise ValueError(f'Expecting ndim 2 for obs, got {len(obs.shape)}')
    obs = jax.tree_util.tree_map(lambda x: x[batch_idx], obs)

  # Shape: (obs_A,) -> ()
  obs = jax.tree_map(lambda x: x[obj_idx], obs)
  if obs.shape:
    raise ValueError(f'Expecting shape () for obs, got {obs.shape}')

  viz_config = (
      utils.VizConfig() if viz_config is None else utils.VizConfig(**viz_config)
  )
  fig, ax = utils.init_fig_ax(viz_config)

  # 1. Plots trajectory.
  # Shape: (num_objects, num_timesteps).
  traj = obs.trajectory
  # The current timestep index of observation is the last one in time dimension
  # as observation is toward the past.
  timestep = traj.num_timesteps - 1
  indices = np.arange(traj.num_objects) if viz_config.show_agent_id else None

  is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
  plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

  # 2. Plots road graph elements.
  # Shape: (num_points,)
  plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

  # Shape: (num_traffic_lights, num_timesteps).
  plot_traffic_light_signals_as_points(
      ax, obs.traffic_lights, timestep, verbose=False
  )

  ax.quiver(0, 0, heading[..., 0], heading[..., 1], color='cyan')

  # 3. Gets np img, centered on selected agent's current location.
  # Shape (num_objects, 2).
  current_xy = traj.xy[:, timestep, :]
  if viz_config.center_agent_idx == -1:
    xy = current_xy[obs.metadata.is_sdc]
  else:
    xy = current_xy[viz_config.center_agent_idx]
  origin_x, origin_y = xy[0, :2]
  ax.axis((
      origin_x - viz_config.back_x,
      origin_x + viz_config.front_x,
      origin_y - viz_config.back_y,
      origin_y + viz_config.front_y,
  ))
  
  return utils.img_from_fig(fig)