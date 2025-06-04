from intersection_env import IntersectionEnv
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np


class CustomIntersectionEnv(IntersectionEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config)
        self.max_speed = 45         # Target max speed (for any speed‐related terms)
        self.render_mode = render_mode
        self.safe_spacing = 15        # Threshold for rewarding good spacing
        self.emergency_distance = 4.0
        self._max_steps = 450
        self.info = None

        # Will hold the Euclidean distance to the goal from the previous step
        self.prev_dist_to_goal = None


    def step(self, action):
        """
        1) Call parent class to advance the simulation.
        2) Check for dangerously close vehicles (distance < min_safe_distance).
            If so, assign a large negative reward and terminate immediately.
        3) Otherwise, compute our custom reward with _reward(action).
        """
        obs, _, done, truncated, self.info = super().step(action)
        self.steps += 1
        if self.steps >= self._max_steps and not done:
            truncated = True

        # Check for dangerous proximity to other vehicles
        # for other_vehicle in self.road.vehicles:
        #     if other_vehicle is not self.vehicle:
        #         distance = np.linalg.norm(self.vehicle.position - other_vehicle.position)
        #         if distance < self.emergency_distance:
        #             reward = -20
        #             self.info['dangerous_proximity'] = True
        #             return obs, reward, done, truncated, self.info
        
        # Compute custom reward
        custom_reward = self._reward(action)
        return obs, custom_reward, done, truncated, self.info

    def _reward(self, action):
        ego = self.vehicle
        reward = 0.0

        # 2) Destination bonus
        if self._reached_destination():
            self.info['arrived'] = True
            return +20.0
        
        # 1) Crash or off‐road penalty
        if ego.crashed:
            self.info['crashed'] = True
            if self.vehicle.speed == 0:
                return 0
            return -20.0
        
        if ego.lane is None:
            return -20.0

        # 3) Forward progress
        dest = ego.destination
        dist = np.linalg.norm(ego.position - dest)
        # Use max_dist_to_goal (set in reset) to normalize
        # reward += (1.0 / 100.0) * (self.max_dist_to_goal - dist)

        # 4) Speed penalty if above max_speed
        if ego.speed > self.max_speed:
            reward -= 0.1 * (ego.speed - self.max_speed)
            
        # 5) Spacing
        min_dist = float("inf")
        for other in self.road.vehicles:
            if other is ego:
                continue
            d = np.linalg.norm(ego.position - other.position)
            if d < min_dist:
                min_dist = d

        # if min_dist >= self.safe_spacing:
        #     reward += 0.1
        if min_dist <= self.safe_spacing and self.vehicle.speed < 2:
            reward += 1
        elif min_dist <= self.emergency_distance and self.vehicle.speed >= 2.5:
            reward -= 1
        # elif min_dist < self.emergency_distance:
        #     reward -= 0.5

        return reward

    def _is_terminated(self):
        return self.vehicle.crashed or self._reached_destination()

    def _is_truncated(self):
        return self.steps >= 500

    def _reached_destination(self):
        dest_x, dest_y = self.vehicle.destination
        x, y = self.vehicle.position
        return abs(dest_x - x) < 5.0 and abs(dest_y - y) < 5.0

    def reset(self, *, seed=None, options=None):
        """
        Override reset to accept (seed, options) keywords, 
        call the parent reset, then compute max_dist_to_goal.
        """
        # 1) Call parent with the exact signature Gym expects:
        obs, info = super().reset(seed=seed, options=options)
        self.steps = 0
        self.info = info

        # 2) After resetting, compute the maximum distance to goal for normalization.
        ego = self.vehicle
        dx = ego.destination[0] - ego.position[0]
        dy = ego.destination[1] - ego.position[1]
        self.max_dist_to_goal = np.linalg.norm([dx, dy])

        # 3) Return exactly what super().reset returned
        return obs, info

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate the road with:
         1) Random background traffic
         2) (Optional) A single challenger
         3) The ego (ControlledVehicle)
        """
        # Configure other vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7.0
        vehicle_type.COMFORT_ACC_MAX = 6.0
        vehicle_type.COMFORT_ACC_MIN = -3.0

        # 1) Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (self.road.act(), self.road.step(1 / self.config["simulation_frequency"]))
                for _ in range(self.config["simulation_frequency"])
            ]

        # 2) (Optional) Challenger vehicle
        self._spawn_vehicle(
            60,
            spawn_probability=1,
            go_straight=True,
            position_deviation=0.1,
            speed_deviation=0,
        )

        # 3) Ego (ControlledVehicle)
        self.controlled_vehicles = []
        for ego_id in range(self.config["controlled_vehicles"]):
            # Always spawn on south approach ("o0" → "ir0")
            ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
            destination = self.config["destination"] or "o3"
            initial_position = ego_lane.position(60, 0)

            ego_vehicle = ControlledVehicle(
                self.road,
                initial_position,
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            ego_vehicle.plan_route_to(destination)
            if hasattr(ego_vehicle, "speed_index"):
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(
                    ego_vehicle.speed_index
                )

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)

            # Remove any car that spawned too close (< 20m) to ego
            for v in list(self.road.vehicles):
                if v is not ego_vehicle and np.linalg.norm(
                    v.position - ego_vehicle.position
                ) < 20.0:
                    self.road.vehicles.remove(v)
