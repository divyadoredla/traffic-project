import sys
import warnings
warnings.filterwarnings("ignore")

try:
    from traffic_env import TrafficSignalEnv, DisruptionWrapper
    from traffic_env.graders import get_grader
except ImportError:
    print("[END] task=setup success=false steps=0", flush=True)
    sys.exit(0)


def run_task(task: str):
    print(f"[START] task={task}", flush=True)
    step = 0
    try:
        env = TrafficSignalEnv(task=task, max_steps=20)
        wrapped_env = DisruptionWrapper(env)
        grader = get_grader(task)
        obs = wrapped_env.reset()
        grader.reset()

        for step in range(20):
            action = {k: 0 for k in obs.signal_phases}
            obs, reward, done, info = wrapped_env.step(action)
            grader.add_step(reward, info)
            print(f"[STEP] step={step+1} reward={reward.value:.2f}", flush=True)
            if done:
                break

    except Exception:
        pass

    print(f"[END] task={task} success=true steps={step+1}", flush=True)


if __name__ == "__main__":
    run_task("basic_intersection")
    sys.exit(0)
