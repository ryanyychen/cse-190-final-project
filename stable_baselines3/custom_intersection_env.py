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
        self.max_speed = 35         # Target max speed (for any speed‐related terms)
        self.render_mode = render_mode
        self.safe_spacing = 5.0        # Threshold for rewarding good spacing
        self.emergency_distance = 2.0

        # Will hold the Euclidean distance to the goal from the previous step
        self.prev_dist_to_goal = None

    def reset(self, **kwargs):
        """
        Override reset so we can re‐initialize prev_dist_to_goal.
        """
        obs, info = super().reset(**kwargs)
        dest_pos = np.array(self.vehicle.destination)
        ego_pos = np.array(self.vehicle.position)
        self.prev_dist_to_goal = np.linalg.norm(dest_pos - ego_pos)
        return obs, info

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
        """
        Custom reward function that:
         1) Penalizes crashing
         2) Penalizes going off‐road
         3) Rewards reaching the destination
         4) Rewards forward progress (Δ distance to goal)
         5) Rewards staying on the planned route (small bonus/penalty)
         6) Penalty for exceeding max speed
         7) Penalty for negative acceleration (accelerating backwards)
         8) Penalty for negative velocity (going in reverse)
         9) Positive reward for maintaining forward speed
        10) Positive reward for keeping safe spacing from other vehicles
        """
        reward = 0.0
        ego = self.vehicle

        # 1) Crash penalty
        if ego.crashed:
            reward -= 50

        # 2) Off‐road penalty (ego.lane is None when off any valid lane)
        if ego.lane is None:
            reward -= 5

        # 3) Reward for successfully reaching destination
        if self._reached_destination():
            reward += 50.0

        # 5) Route‐following reward or small penalty
        on_route_bonus = 0.0
        try:
            planned_route = ego.route  # deque of (node_id, lane_id, offset)
            current_lane_id = (
                ego.lane_index[1] if ego.lane_index and len(ego.lane_index) > 1 else None
            )
            route_lane_ids = [step[1] for step in planned_route]
            if current_lane_id in route_lane_ids:
                on_route_bonus = 0.1
            else:
                on_route_bonus = -0.5
        except Exception:
            on_route_bonus = 0.0
        reward += on_route_bonus

        # 6) Penalty for exceeding max speed
        if ego.speed > self.max_speed:
            speed_excess = ego.speed - self.max_speed
            reward -= 1.0 * speed_excess

        # 10) Positive reward for keeping safe spacing from other vehicles
        #     Compute minimum distance to any other vehicle
        min_dist = float("inf")
        for other in self.road.vehicles:
            if other is not ego:
                dist_i = np.linalg.norm(ego.position - other.position)
                if dist_i < min_dist:
                    min_dist = dist_i

        # If the minimum distance is above the safe_spacing threshold, give a bonus
        if min_dist > self.safe_spacing:
            spacing_bonus = 1.0 * (min_dist / self.safe_spacing)
            reward += spacing_bonus

        # EMERGENCY‐STOP bonus: if min_dist < emergency_distance, reward speed ≈ 0
        if min_dist < self.emergency_distance:
            # if ego is almost stopped, give a high bonus
            if abs(ego.speed) < 0.1:
                reward += 2
            else:
                # if ego still moving when extremely close, stronger penalty
                reward -= 2 * (abs(ego.speed) / self.max_speed)

        return reward

    def _is_terminated(self):
        """
        End the episode if ego crashed or reached destination.
        (Dangerous proximity already handled in step().)
        """
        return self.vehicle.crashed or self._reached_destination()

    def _is_truncated(self):
        """
        Cap the episode length at 500 steps.
        """
        return self.steps >= 500

    def _reached_destination(self):
        """
        Return True if ego is within a small threshold of its destination.
        """
        dest_x, dest_y = self.vehicle.destination
        threshold = 5.0  # meters
        vehicle_x, vehicle_y = self.vehicle.position
        return (
            abs(dest_x - vehicle_x) < threshold
            and abs(dest_y - vehicle_y) < threshold
        )

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
