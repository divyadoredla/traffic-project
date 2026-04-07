import numpy as np
from typing import Dict, Any, Tuple
from .environment import TrafficSignalEnv, TrafficObservation, TrafficReward


class DisruptionWrapper:
    """
    Wrapper that injects real-world disruptions into the traffic environment
    """
    
    def __init__(self, env: TrafficSignalEnv, disruption_config: Dict[str, Any] = None):
        self.env = env
        self.disruption_config = disruption_config or self._default_config()
        self.active_disruptions = {}
        self.disruption_timers = {}
        
    def _default_config(self) -> Dict[str, Any]:
        """Default disruption configuration"""
        return {
            "lane_blockage": {
                "probability": self.env.disruption_probability,
                "duration_range": (10, 30),
                "severity": 1.0
            },
            "demand_spike": {
                "probability": 0.05,
                "duration_range": (5, 15),
                "multiplier_range": (2.0, 4.0)
            },
            "sensor_failure": {
                "probability": 0.02,
                "duration_range": (20, 50),
                "noise_level": 0.3
            }
        }
    
    def reset(self) -> TrafficObservation:
        """Reset environment and disruption state"""
        self.active_disruptions = {}
        self.disruption_timers = {}
        return self.env.reset()
    
    def step(self, action: Dict[str, int]) -> Tuple[TrafficObservation, TrafficReward, bool, Dict[str, Any]]:
        """Execute step with disruption injection"""
        # Inject new disruptions
        self._inject_disruptions()
        
        # Update existing disruptions
        self._update_disruptions()
        
        # Apply disruptions to environment
        self._apply_disruptions()
        
        # Execute environment step
        obs, reward, done, info = self.env.step(action)
        
        # Modify reward based on disruption handling
        modified_reward = self._modify_reward(reward)
        
        # Add disruption info
        info["disruptions"] = {
            "active": len(self.active_disruptions),
            "types": list(self.active_disruptions.keys())
        }
        
        return obs, modified_reward, done, info
    
    def _inject_disruptions(self):
        """Randomly inject new disruptions"""
        for disruption_type, config in self.disruption_config.items():
            if np.random.random() < config["probability"]:
                self._create_disruption(disruption_type, config)
    
    def _create_disruption(self, disruption_type: str, config: Dict[str, Any]):
        """Create a new disruption"""
        disruption_id = f"{disruption_type}_{len(self.active_disruptions)}"
        
        if disruption_type == "lane_blockage":
            # Block a random lane
            intersection_id = np.random.randint(self.env.num_intersections)
            lane_id = np.random.randint(self.env.num_lanes_per_intersection)
            target_lane = f"intersection_{intersection_id}_lane_{lane_id}"
            
            self.active_disruptions[disruption_id] = {
                "type": disruption_type,
                "target": target_lane,
                "severity": config["severity"],
                "duration": np.random.randint(*config["duration_range"])
            }
            
        elif disruption_type == "demand_spike":
            # Increase vehicle arrival rate
            intersection_id = np.random.randint(self.env.num_intersections)
            multiplier = np.random.uniform(*config["multiplier_range"])
            
            self.active_disruptions[disruption_id] = {
                "type": disruption_type,
                "target": f"intersection_{intersection_id}",
                "multiplier": multiplier,
                "duration": np.random.randint(*config["duration_range"])
            }
            
        elif disruption_type == "sensor_failure":
            # Add noise to observations
            self.active_disruptions[disruption_id] = {
                "type": disruption_type,
                "noise_level": config["noise_level"],
                "duration": np.random.randint(*config["duration_range"])
            }
        
        self.disruption_timers[disruption_id] = 0
    
    def _update_disruptions(self):
        """Update disruption timers and remove expired ones"""
        expired_disruptions = []
        
        for disruption_id in self.active_disruptions:
            self.disruption_timers[disruption_id] += 1
            
            if (self.disruption_timers[disruption_id] >= 
                self.active_disruptions[disruption_id]["duration"]):
                expired_disruptions.append(disruption_id)
        
        # Remove expired disruptions
        for disruption_id in expired_disruptions:
            del self.active_disruptions[disruption_id]
            del self.disruption_timers[disruption_id]
    
    def _apply_disruptions(self):
        """Apply active disruptions to the environment"""
        # Reset disruption flags
        for lane_id in self.env.disruptions:
            self.env.disruptions[lane_id] = False
        
        # Apply active disruptions
        for disruption_id, disruption in self.active_disruptions.items():
            if disruption["type"] == "lane_blockage":
                target_lane = disruption["target"]
                if target_lane in self.env.disruptions:
                    self.env.disruptions[target_lane] = True
                    
            elif disruption["type"] == "demand_spike":
                # This will be handled during traffic simulation
                pass
                
            elif disruption["type"] == "sensor_failure":
                # Add noise to vehicle counts
                for lane_id in self.env.vehicle_counts:
                    noise = np.random.normal(0, disruption["noise_level"] * 5)
                    self.env.vehicle_counts[lane_id] = max(0, 
                        int(self.env.vehicle_counts[lane_id] + noise))
    
    def _modify_reward(self, original_reward: TrafficReward) -> TrafficReward:
        """Modify reward to account for disruption handling"""
        # Bonus for maintaining performance during disruptions
        disruption_bonus = 0.0
        if len(self.active_disruptions) > 0:
            # Small bonus for each active disruption handled well
            if original_reward.value > 0:
                disruption_bonus = 0.1 * len(self.active_disruptions)
        
        modified_value = original_reward.value + disruption_bonus
        modified_value = np.clip(modified_value, -1.0, 1.0)
        
        # Update components
        components = original_reward.components.copy()
        components["disruption_bonus"] = disruption_bonus
        
        return TrafficReward(
            value=modified_value,
            components=components
        )
    
    def state(self) -> Dict[str, Any]:
        """Return current state including disruptions"""
        env_state = self.env.state()
        env_state["active_disruptions"] = self.active_disruptions.copy()
        env_state["disruption_timers"] = self.disruption_timers.copy()
        return env_state
    
    def close(self):
        """Clean up resources"""
        self.env.close()