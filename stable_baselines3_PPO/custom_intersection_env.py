from highway_env.envs.intersection_env import IntersectionEnv
import numpy as np

class CustomIntersectionEnv(IntersectionEnv):

    def _reward(self, action):
        reward = 0.0
        
        if self.vehicle.crashed:
            reward -= 5.0
        
        # Reward for successfully reaching destination
        if self._reached_destination():
            reward += 35.0  # Large bonus for success
        
        # is the ego vehicle in the intersection?
        in_intersection = "ir" in self.vehicle.lane_index[0] if self.vehicle.lane else False

        # Count vehicles in intersection
        vehicles_in_intersection = 0
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:
                if other_vehicle.lane and "ir" in other_vehicle.lane_index[0]:
                    vehicles_in_intersection += 1

        safe_distance_reward = 0
        for other_vehicle in self.road.vehicles:
            if other_vehicle is not self.vehicle:
                distance = np.linalg.norm(self.vehicle.position - other_vehicle.position)
                # Calculate relative velocity (positive means approaching)
                # rel_velocity = np.dot(
                #     other_vehicle.velocity - self.vehicle.velocity,
                #     (other_vehicle.position - self.vehicle.position) / (distance + 1e-6)
                # )

                min_safe_distance = 10
                emergency_stop_distance = 5
                if distance < min_safe_distance:
                    if self.vehicle.speed < 0.5:
                        safe_distance_reward += 5.0  # reward for full stop near car

                    # Stronger penalty for unsafe distances
                    safe_distance_reward -= 1.5 * (min_safe_distance - distance) / min_safe_distance
                    
                    # Additional penalty for high speed at close distances
                    if self.vehicle.speed > 2.0 and distance < emergency_stop_distance:
                        safe_distance_reward -= 1.5 * (self.vehicle.speed / 10.0)
                    
                    # Reward for stopping when too close
                    if distance < emergency_stop_distance and self.vehicle.speed < 1.0:
                        safe_distance_reward += 5.0
                    
                    break
        
        # Strong penalty for entering intersection when other vehicles are present
        # if in_intersection and vehicles_in_intersection > 0 and self.vehicle.speed > 2.0:
        #     safe_distance_reward -= 3.0

        reward += safe_distance_reward

        # reward -= 0.1 * self.vehicle.speed  # discourage excessive speed

        return reward
       

    def _is_terminated(self):
        # End the episode if crashed or reached destination
        return self.vehicle.crashed or self._reached_destination()

    def _is_truncated(self):
        # Episode length cap (adjustable as needed)
        return self.steps >= 500

    def _reached_destination(self):
        dest_x = self.vehicle.destination[0]  # Adjust according to intersection layout (leftward position)
        dest_y = self.vehicle.destination[1]  # Tolerance in y-axis direction
        threshold = 5

        vehicle_x, vehicle_y = self.vehicle.position

        # Vehicle must have moved significantly leftwards and be within lane vertically
        return abs(dest_x - vehicle_x) < threshold and abs(dest_y - vehicle_y) < threshold