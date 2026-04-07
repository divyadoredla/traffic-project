import numpy as np
from typing import List, Dict, Any
from .environment import TrafficReward


class BaseGrader:
    """Base class for task graders"""
    
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.episode_rewards = []
        self.episode_info = []
    
    def reset(self):
        """Reset grader state for new episode"""
        self.episode_rewards = []
        self.episode_info = []
    
    def add_step(self, reward: TrafficReward, info: Dict[str, Any]):
        """Add step data for grading"""
        self.episode_rewards.append(reward.value)
        self.episode_info.append(info)
    
    def grade(self) -> float:
        """Calculate final grade (0.0 to 1.0)"""
        raise NotImplementedError


class BasicIntersectionGrader(BaseGrader):
    """Grader for basic intersection task (easy difficulty)"""
    
    def __init__(self):
        super().__init__("basic_intersection")
        self.target_throughput = 0.7
        self.max_waiting_time = 50.0
    
    def grade(self) -> float:
        """Grade based on throughput and waiting time"""
        if not self.episode_rewards:
            return 0.0
        
        # Average reward over episode
        avg_reward = np.mean(self.episode_rewards)
        
        # Throughput performance
        avg_waiting_times = [info.get("average_waiting_time", 100) 
                           for info in self.episode_info]
        avg_waiting = np.mean(avg_waiting_times) if avg_waiting_times else 100
        
        # Normalize metrics
        reward_score = max(0, (avg_reward + 1) / 2)  # Convert from [-1,1] to [0,1]
        waiting_score = max(0, 1 - avg_waiting / self.max_waiting_time)
        
        # Stability bonus (consistent performance)
        reward_std = np.std(self.episode_rewards)
        stability_bonus = max(0, 0.2 * (1 - reward_std))
        
        # Final score
        final_score = 0.5 * reward_score + 0.3 * waiting_score + stability_bonus
        return min(1.0, final_score)


class MultiIntersectionGrader(BaseGrader):
    """Grader for multi-intersection task (medium difficulty)"""
    
    def __init__(self):
        super().__init__("multi_intersection")
        self.target_throughput = 0.6
        self.max_disruptions = 5
    
    def grade(self) -> float:
        """Grade based on coordination and disruption handling"""
        if not self.episode_rewards:
            return 0.0
        
        # Performance metrics
        avg_reward = np.mean(self.episode_rewards)
        
        # Disruption handling
        disruption_counts = [info.get("active_disruptions", 0) 
                           for info in self.episode_info]
        avg_disruptions = np.mean(disruption_counts) if disruption_counts else 0
        
        # Recovery performance (reward improvement after disruptions)
        recovery_scores = []
        for i in range(1, len(self.episode_rewards)):
            if (disruption_counts[i-1] > 0 and 
                self.episode_rewards[i] > self.episode_rewards[i-1]):
                recovery_scores.append(1.0)
            elif disruption_counts[i-1] > 0:
                recovery_scores.append(0.0)
        
        recovery_rate = np.mean(recovery_scores) if recovery_scores else 0.5
        
        # Normalize metrics
        reward_score = max(0, (avg_reward + 1) / 2)
        disruption_score = max(0, 1 - avg_disruptions / self.max_disruptions)
        
        # Final score with emphasis on disruption handling
        final_score = (0.4 * reward_score + 
                      0.3 * disruption_score + 
                      0.3 * recovery_rate)
        return min(1.0, final_score)


class CityNetworkGrader(BaseGrader):
    """Grader for city network task (hard difficulty)"""
    
    def __init__(self):
        super().__init__("city_network")
        self.target_throughput = 0.5
        self.resilience_threshold = 0.3
    
    def grade(self) -> float:
        """Grade based on system-wide resilience and adaptation"""
        if not self.episode_rewards:
            return 0.0
        
        # Performance under stress
        avg_reward = np.mean(self.episode_rewards)
        
        # Resilience metrics
        disruption_counts = [info.get("active_disruptions", 0) 
                           for info in self.episode_info]
        
        # Performance during high disruption periods
        high_disruption_indices = [i for i, count in enumerate(disruption_counts) 
                                 if count >= 3]
        
        if high_disruption_indices:
            stress_rewards = [self.episode_rewards[i] for i in high_disruption_indices]
            stress_performance = np.mean(stress_rewards)
        else:
            stress_performance = avg_reward
        
        # Adaptation speed (how quickly performance recovers)
        adaptation_scores = []
        for i in range(5, len(self.episode_rewards)):
            recent_trend = np.mean(self.episode_rewards[i-4:i+1])
            if recent_trend > self.episode_rewards[i-5]:
                adaptation_scores.append(1.0)
            else:
                adaptation_scores.append(0.0)
        
        adaptation_rate = np.mean(adaptation_scores) if adaptation_scores else 0.5
        
        # System stability (low variance in performance)
        stability = 1.0 / (1.0 + np.std(self.episode_rewards))
        
        # Normalize metrics
        reward_score = max(0, (avg_reward + 1) / 2)
        stress_score = max(0, (stress_performance + 1) / 2)
        
        # Final score emphasizing resilience
        final_score = (0.3 * reward_score + 
                      0.4 * stress_score + 
                      0.2 * adaptation_rate + 
                      0.1 * stability)
        return min(1.0, final_score)


def get_grader(task_name: str) -> BaseGrader:
    """Factory function to get appropriate grader"""
    graders = {
        "basic_intersection": BasicIntersectionGrader,
        "multi_intersection": MultiIntersectionGrader,
        "city_network": CityNetworkGrader
    }
    
    if task_name not in graders:
        raise ValueError(f"Unknown task: {task_name}")
    
    return graders[task_name]()