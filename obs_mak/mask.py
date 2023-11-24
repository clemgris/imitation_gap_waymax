from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp
from typing import Any


class ObsMask(ABC):

    @abstractmethod
    def mask_fun(self, *args: Any, **kwds: Any) -> Any:
        pass

    @abstractmethod
    def plot_mask_fun(self, *args: Any, **kwds: Any) -> None:
        pass

@dataclass
class DistanceObsMask(ObsMask):
    radius: float

    def mask_fun(self, obj_x, obj_y):
        squared_distance = obj_x**2 + obj_y**2

        return squared_distance <= self.radius**2
    
    def plot_mask_fun(self, ax, center) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        ax.plot(x, y)

@dataclass
class ConicObsMask(ObsMask):
    radius: float
    angle: float

    def mask_fun(self, obj_x, obj_y):
        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition = (- self.angle / 2 <= obj_angle) & (obj_angle <= self.angle / 2)
        radius_condition = squared_distance <= self.radius**2

        return angle_condition & radius_condition
    
    def plot_mask_fun(self, ax, center, color='b') -> None:
        theta = jnp.linspace(- self.angle/2, self.angle/2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- self.angle/2)
        y1 = center[1] + self.radius * jnp.sin(- self.angle/2)

        x2 = center[0] + self.radius * jnp.cos(self.angle/2)
        y2 = center[1] + self.radius * jnp.sin(self.angle/2)

        ax.plot([0, x1], [0, y1], c=color)
        ax.plot([0, x2], [0, y2], c=color)
        ax.plot(x, y, c=color)