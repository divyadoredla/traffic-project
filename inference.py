import os
import warnings
warnings.filterwarnings("ignore")

from traffic_env import TrafficSignalEnv, DisruptionWrapper
from traffic_env.graders import get_grader

MAX_STEPS = 20


def run_task(task: str):
    print(f"[START] task={task}", flush=True)

    env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
    wrapped_env = DisruptionWrapper(env)
    grader = get_grader(task)
    obs = wrapped_env.reset()
    grader.reset()

    step = 0
    for step in range(MAX_STEPS):
        action = {k: 0 for k in obs.signal_phases}
        obs, reward, done, info = wrapped_env.step(action)
        grader.add_step(reward, info)
        print(f"[STEP] step={step} reward={reward.value}", flush=True)
        if done:
            break

    wrapped_env.close()
    print(f"[END] task={task} success=true steps={step}", flush=True)


if __name__ == "__main__":
    try:
        run_task("basic_intersection")
    except Exception as e:
        print(f"[END] task=basic_intersection success=false", flush=True)
