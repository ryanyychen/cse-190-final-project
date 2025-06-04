from intersection_env import IntersectionEnv
from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import AbstractLane, CircularLane, LineType, StraightLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.controller import ControlledVehicle
import numpy as np


class SimpleIntersectionEnv(IntersectionEnv):
    def __init__(self, config=None, render_mode=None):
        super().__init__(config)
        self.render_mode = render_mode
        self._max_steps = 800


    def step(self, action):
        """
        1) Call parent class to advance the simulation.
        2) Check for dangerously close vehicles (distance < min_safe_distance).
            If so, assign a large negative reward and terminate immediately.
        3) Otherwise, compute our custom reward with _reward(action).
        """
        obs, _, done, truncated, self.info = super().step(action)
        custom_reward = self._reward(action)
        self.steps += 1
        
        if self.steps >= self._max_steps and not done:
            truncated = True

        return obs, custom_reward, done, truncated, self.info

    def _reward(self, action):
        ego = self.vehicle
        if ego.crashed:
            self.info['crashed'] = True
            return -10.0
        if ego.lane is None:
            return -10.0
        if self._reached_destination():
            self.info['arrived'] = True
            return +10.0
        
        return 0


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
