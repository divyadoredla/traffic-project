import numpy as np
from typing import Dict, Any, Tuple, Optional
from pydantic import BaseModel
import gymnasium as gym
from gymnasium import spaces


class TrafficObservation(BaseModel):
    """Typed observation model for traffic state"""
    vehicle_counts: Dict[str, int]  # Vehicles per lane
    signal_phases: Dict[str, int]   # Current signal phase per intersection
    waiting_times: Dict[str, float]  # Average waiting time per lane
    disruptions: Dict[str, bool]    # Active disruptions per lane
    throughput: float               # Recent throughput rate
    
    class Config:
        arbitrary_types_allowed = True


class TrafficAction(BaseModel):
    """Typed action model for signal control"""
    intersection_phases: Dict[str, int]  # Phase selection per intersection
    
    class Config:
        arbitrary_types_allowed = True


class TrafficReward(BaseModel):
    """Typed reward model"""
    value: float
    components: Dict[str, float]  # Breakdown of reward components
    
    class Config:
        arbitrary_types_allowed = True


class TrafficSignalEnv:
    """
    OpenEnv-compliant traffic signal control environment with incident resilience
    """
    
    def __init__(self, task: str = "basic_intersection", max_steps: int = 200):
        self.task = task
        self.max_steps = max_steps
        self.current_step = 0
        
        # Task-specific configuration
        if task == "basic_intersection":
            self.num_intersections = 1
            self.num_lanes_per_intersection = 4
            self.disruption_probability = 0.1
        elif task == "multi_intersection":
            self.num_intersections = 3
            self.num_lanes_per_intersection = 4
            self.disruption_probability = 0.2
        elif task == "city_network":
            self.num_intersections = 6
            self.num_lanes_per_intersection = 6
            self.disruption_probability = 0.3
        else:
            raise ValueError(f"Unknown task: {task}")
            
        # Initialize state
        self.vehicle_counts = {}
        self.signal_phases = {}
        self.waiting_times = {}
        self.disruptions = {}
        self.throughput_history = []
        
        # Action and observation spaces
        self.action_space = spaces.Dict({
            f"intersection_{i}": spaces.Discrete(4)  # 4 signal phases
            for i in range(self.num_intersections)
        })
        
        self.observation_space = spaces.Dict({
            "vehicle_counts": spaces.Box(
                low=0, high=100, 
                shape=(self.num_intersections * self.num_lanes_per_intersection,),
                dtype=np.int32
            ),
            "signal_phases": spaces.Box(
                low=0, high=3,
                shape=(self.num_intersections,),
                dtype=np.int32
            ),
            "waiting_times": spaces.Box(
                low=0, high=300,
                shape=(self.num_intersections * self.num_lanes_per_intersection,),
                dtype=np.float32
            ),
            "disruptions": spaces.Box(
                low=0, high=1,
                shape=(self.num_intersections * self.num_lanes_per_intersection,),
                dtype=np.int32
            ),
            "throughput": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
    def reset(self) -> TrafficObservation:
        """Reset environment to initial state"""
        self.current_step = 0
        
        # Initialize vehicle counts
        for i in range(self.num_intersections):
            for j in range(self.num_lanes_per_intersection):
                lane_id = f"intersection_{i}_lane_{j}"
                self.vehicle_counts[lane_id] = np.random.randint(5, 20)
                self.waiting_times[lane_id] = np.random.uniform(0, 10)
                self.disruptions[lane_id] = False
                
        # Initialize signal phases
        for i in range(self.num_intersections):
            self.signal_phases[f"intersection_{i}"] = 0
            
        self.throughput_history = [0.5]  # Start with moderate throughput
        
        return self._get_observation()
    
    def step(self, action: Dict[str, int]) -> Tuple[TrafficObservation, TrafficReward, bool, Dict[str, Any]]:
        """Execute one environment step"""
        self.current_step += 1
        
        # Update signal phases based on action
        for intersection_id, phase in action.items():
            if intersection_id in self.signal_phases:
                self.signal_phases[intersection_id] = phase
        
        # Simulate traffic dynamics
        self._simulate_traffic()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Additional info
        info = {
            "step": self.current_step,
            "active_disruptions": sum(self.disruptions.values()),
            "average_waiting_time": np.mean(list(self.waiting_times.values())),
            "last_action_error": None
        }
        
        return self._get_observation(), reward, done, info
    
    def _simulate_traffic(self):
        """Simulate traffic flow and disruptions"""
        # Update vehicle counts based on signal phases
        for i in range(self.num_intersections):
            intersection_id = f"intersection_{i}"
            current_phase = self.signal_phases[intersection_id]
            
            for j in range(self.num_lanes_per_intersection):
                lane_id = f"intersection_{i}_lane_{j}"
                
                # Traffic flow based on signal phase and disruptions
                if not self.disruptions[lane_id]:
                    if j % 2 == current_phase % 2:  # Green light for this direction
                        # Vehicles can pass through
                        vehicles_passed = min(self.vehicle_counts[lane_id], 
                                            np.random.randint(3, 8))
                        self.vehicle_counts[lane_id] -= vehicles_passed
                        self.waiting_times[lane_id] *= 0.8  # Reduce waiting time
                    else:
                        # Red light - vehicles accumulate
                        self.waiting_times[lane_id] += 1.0
                else:
                    # Lane blocked - severe congestion
                    self.waiting_times[lane_id] += 2.0
                
                # New vehicles arrive
                new_vehicles = np.random.poisson(2)
                self.vehicle_counts[lane_id] += new_vehicles
                self.vehicle_counts[lane_id] = min(self.vehicle_counts[lane_id], 100)
        
        # Calculate throughput
        total_vehicles = sum(self.vehicle_counts.values())
        max_vehicles = self.num_intersections * self.num_lanes_per_intersection * 100
        current_throughput = 1.0 - (total_vehicles / max_vehicles)
        self.throughput_history.append(current_throughput)
        
        # Keep only recent history
        if len(self.throughput_history) > 10:
            self.throughput_history.pop(0)
    
    def _calculate_reward(self) -> TrafficReward:
        """Calculate reward based on traffic performance"""
        # Throughput component (higher is better)
        throughput_reward = np.mean(self.throughput_history)
        
        # Waiting time component (lower is better)
        avg_waiting = np.mean(list(self.waiting_times.values()))
        waiting_reward = max(0, 1.0 - avg_waiting / 100.0)
        
        # Disruption handling component
        active_disruptions = sum(self.disruptions.values())
        disruption_penalty = -0.1 * active_disruptions
        
        # Combine components
        total_reward = 0.4 * throughput_reward + 0.4 * waiting_reward + disruption_penalty
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        return TrafficReward(
            value=total_reward,
            components={
                "throughput": throughput_reward,
                "waiting_time": waiting_reward,
                "disruption_penalty": disruption_penalty
            }
        )
    
    def _get_observation(self) -> TrafficObservation:
        """Get current observation"""
        return TrafficObservation(
            vehicle_counts=self.vehicle_counts.copy(),
            signal_phases=self.signal_phases.copy(),
            waiting_times=self.waiting_times.copy(),
            disruptions=self.disruptions.copy(),
            throughput=np.mean(self.throughput_history)
        )
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        return {
            "step": self.current_step,
            "vehicle_counts": self.vehicle_counts,
            "signal_phases": self.signal_phases,
            "waiting_times": self.waiting_times,
            "disruptions": self.disruptions,
            "throughput_history": self.throughput_history
        }
    
    def close(self):
        """Clean up environment resources"""
        pass