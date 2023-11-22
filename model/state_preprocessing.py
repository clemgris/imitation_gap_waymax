from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from typing import Any, Dict

from model.config import UNVALID_MASK_VALUE
from waymax import datatypes

from utils.observation import last_sdc_observation_for_current_sdc_from_state

class Extractor(ABC):

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
        
    @abstractmethod
    def init_x(self):
        pass

@dataclass
class ExtractXY(Extractor):
    config: Dict

    def __call__(self, state):
        obs = datatypes.sdc_observation_from_state(state,
                                                   roadgraph_top_k=self.config['roadgraph_top_k'])
        traj = obs.trajectory.xy
        valid = obs.trajectory.valid[..., None]
        masked_traj = jnp.where(valid, traj, UNVALID_MASK_VALUE * jnp.ones_like(traj))

        return {'xy': masked_traj}
    
    def init_x(self,):
        return  ({'xy': jnp.zeros((1, self.config["num_envs"], self.config['max_num_obj'], 2))},
                 jnp.zeros((1, self.config["num_envs"]), dtype=bool),
                 )
    
@dataclass
class ExtractXYGoal(Extractor):
    config: Dict

    def __call__(self, state): 
        
        # Last obs of the log in the current SDC pos referential
        last_sdc_pos = last_sdc_observation_for_current_sdc_from_state(state)

        # Get the last log pos of the SDC
        _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
        sdc_xy = jnp.take_along_axis(last_sdc_pos.trajectory.xy[..., 0, :], sdc_idx[..., None, None], axis=-2)
        
        # Mask if no SDC
        mask = jnp.any(state.object_metadata.is_sdc)[..., None, None]
        # Extract batched proxy goal
        proxy_goal = sdc_xy * mask

        return {"xy": ExtractXY(self.config)(state)['xy'],
                "proxy_goal": proxy_goal}

    def init_x(self):

        return(
            {'xy': jnp.zeros((1, self.config["num_envs"], self.config['max_num_obj'], 2)),
            'proxy_goal': jnp.zeros((1, self.config["num_envs"], 2))},
            jnp.zeros((1, self.config["num_envs"]), dtype=bool),
        )