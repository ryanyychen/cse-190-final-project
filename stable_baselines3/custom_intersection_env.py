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
        self.max_speed = 15.0  # Set target max speed to 15
        self.render_mode = render_mode
        self.last_speed = 0.0
        self.min_safe_distance = 2.0  # Minimum safe distance to other vehicles
        
    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        
        # Check for dangerous proximity to other vehicles
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - other_vehicle.position)
                if distance < self.min_safe_distance:
                    # Immediate termination with large negative reward
                    reward = -50.0
                    done = True
                    info['dangerous_proximity'] = True
                    return obs, reward, done, truncated, info
        
        return obs, reward, done, truncated, info

    def _reward(self, action):
        reward = 0.0
        
        if self.vehicle.crashed:
            reward -= 25
        
        # Reward for successfully reaching destination
        if self._reached_destination():
            reward += 50.0  # Increased bonus for success
        
        # is the ego vehicle in the intersection?
        in_intersection = "ir" in self.vehicle.lane_index[0] if self.vehicle.lane else False

        # Count vehicles in intersection
        vehicles_in_intersection = 0
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:
                if other_vehicle.lane and "ir" in other_vehicle.lane_index[0]:
                    vehicles_in_intersection += 1

        # safe_distance_reward = 0
        # for other_vehicle in self.road.vehicles:
        #     if other_vehicle is not self.vehicle:
        #         distance = np.linalg.norm(self.vehicle.position - other_vehicle.position)

        #         min_safe_distance = 10
        #         emergency_stop_distance = 5
        #         if distance < min_safe_distance:
        #             if self.vehicle.speed < 0.5:
        #                 safe_distance_reward += 5.0  # reward for full stop near car

        #             # Stronger penalty for unsafe distances
        #             safe_distance_reward -= 1.0 * (min_safe_distance - distance) / min_safe_distance
                    
        #             # Additional penalty for high speed at close distances
        #             if self.vehicle.speed > 2.0 and distance < emergency_stop_distance:
        #                 safe_distance_reward -= 1.0 * (self.vehicle.speed / self.max_speed)
                    
        #             # Reward for stopping when too close
        #             if distance < min_safe_distance and self.vehicle.speed < 1.0:
        #                 safe_distance_reward += 5.0
                    
        #             break
        
        # reward += safe_distance_reward

        # # Speed-based rewards
        # if self.vehicle.speed <= self.max_speed:
        #     reward += 1.0  # Small reward for staying within speed limit
        # else:
        #     speed_excess = self.vehicle.speed - self.max_speed
        #     reward -= 5.0 * speed_excess  # Penalty for exceeding speed limit

        return reward

    def _is_terminated(self):
        # End the episode if crashed, reached destination, or got too close to another vehicle
        return self.vehicle.crashed or self._reached_destination()

    def _is_truncated(self):
        # Episode length cap (adjustable as needed)
        return self.steps >= 500

    def _reached_destination(self):
        dest_x = self.vehicle.destination[0]
        dest_y = self.vehicle.destination[1]
        threshold = 5

        vehicle_x, vehicle_y = self.vehicle.position

        return abs(dest_x - vehicle_x) < threshold and abs(dest_y - vehicle_y) < threshold

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [
                (
                    self.road.act(),
                    self.road.step(1 / self.config["simulation_frequency"]),
                )
                for _ in range(self.config["simulation_frequency"])
            ]

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            # Start from o0 (south) and go to o3 (east)
            ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
            destination = self.config["destination"] or "o3"
            
            # Set initial position far from destination
            initial_position = ego_lane.position(60, 0)  # Start 60 units away
            
            # Create a controlled vehicle instead of a basic vehicle
            ego_vehicle = ControlledVehicle(
                self.road,
                initial_position,
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60),
            )
            
            # Plan route to destination
            ego_vehicle.plan_route_to(destination)
            
            # Set speed parameters
            if hasattr(ego_vehicle, 'speed_index'):
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            
            # Prevent early collisions
            for v in self.road.vehicles:
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)