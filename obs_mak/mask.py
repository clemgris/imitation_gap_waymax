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

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        return (squared_distance <= self.radius**2) | is_center
    
    def plot_mask_fun(self, ax, center=(0, 0)) -> None:
        theta = jnp.linspace(0, 2 * jnp.pi, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        ax.plot(x, y)

@dataclass
class ConicObsMask(ObsMask):
    radius: float
    angle: float

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)

        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition = (- self.angle / 2 <= obj_angle) & (obj_angle <= self.angle / 2)
        radius_condition = squared_distance <= self.radius**2

        return (angle_condition & radius_condition)| is_center
    
    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
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


@dataclass
class BlindSpotObsMask(ObsMask):
    radius: float
    angle: float

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        assert(self.angle <= jnp.pi / 2)
        is_center = (-eps <= obj_x) & (obj_x <= eps) & \
            (-eps <= obj_y) & (obj_y <= eps)
        
        visible_angle = jnp.pi - 2 * self.angle
        squared_distance = obj_x**2 + obj_y**2

        obj_angle = jnp.arctan2(obj_y, obj_x)
        angle_condition_front = (- jnp.pi / 2 <= obj_angle) & (obj_angle <= jnp.pi / 2)
        angle_condition_back = (- (visible_angle / 2 + 2 * self.angle) >= obj_angle) | (obj_angle >= (visible_angle / 2 + 2 * self.angle))
        radius_condition = squared_distance <= self.radius**2

        return ((angle_condition_front | angle_condition_back) & radius_condition) | is_center
    
    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        
        # Front
        theta = jnp.linspace(- jnp.pi / 2, jnp.pi / 2, 100)

        x = center[0] + self.radius * jnp.cos(theta)
        y = center[1] + self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- jnp.pi / 2)
        y1 = center[1] + self.radius * jnp.sin(- jnp.pi / 2)

        x2 = center[0] + self.radius * jnp.cos(jnp.pi / 2)
        y2 = center[1] + self.radius * jnp.sin(jnp.pi / 2)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

        # Back
        visible_angle = jnp.pi - 2 * self.angle
        theta = jnp.linspace(visible_angle / 2, - visible_angle / 2, 100)

        x = center[0] - self.radius * jnp.cos(theta)
        y = center[1] - self.radius * jnp.sin(theta)

        x1 = center[0] + self.radius * jnp.cos(- (visible_angle / 2 + 2 * self.angle))
        y1 = center[1] + self.radius * jnp.sin(- (visible_angle / 2 + 2 * self.angle))

        x2 = center[0] + self.radius * jnp.cos(visible_angle / 2 + 2 * self.angle)
        y2 = center[1] + self.radius * jnp.sin(visible_angle / 2 + 2 * self.angle)

        ax.plot([center[0], x1], [center[1], y1], c=color)
        ax.plot([center[0], x2], [center[1], y2], c=color)
        ax.plot(x, y, c=color)

@dataclass
class SpeedConicObsMask(ObsMask):
    radius: float
    angle_min: float
    v_max: float
    sdc_v: float

    def mask_fun(self, obj_x, obj_y, eps=1e-3):
        angle = - self.sdc_v * (2 * jnp.pi - self.angle_min) / self.v_max + 2 * jnp.pi

        return ConicObsMask(self.radius, angle).mask_fun(obj_x, obj_y, eps=eps)
    
    def plot_mask_fun(self, ax, center=(0, 0), color='b') -> None:
        angle = - self.sdc_v * (2 * jnp.pi - self.angle_min) / self.v_max + 2 * jnp.pi

        ConicObsMask(self.radius, angle).plot_mask_fun(ax, center=center, color=color)