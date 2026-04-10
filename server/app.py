import os
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from traffic_env import TrafficSignalEnv, DisruptionWrapper
from traffic_env.graders import get_grader

MAX_STEPS = 20
TASKS = ["basic_intersection", "multi_intersection", "city_network"]

app = FastAPI()

envs = {}
graders = {}
steps = {}
last_obs = {}


@app.get("/")
def health():
    return {"status": "ok", "tasks": TASKS}


@app.post("/reset")
def reset(body: dict = None):
    task = (body or {}).get("task", TASKS[0])
    print(f"START: {task}", flush=True)

    env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
    wrapped = DisruptionWrapper(env)
    grader = get_grader(task)
    obs = wrapped.reset()
    grader.reset()

    envs[task] = wrapped
    graders[task] = grader
    steps[task] = 0
    last_obs[task] = obs

    return JSONResponse({
        "task": task,
        "observation": {
            "vehicle_counts": obs.vehicle_counts,
            "signal_phases": obs.signal_phases,
            "waiting_times": obs.waiting_times,
            "disruptions": obs.disruptions,
            "throughput": obs.throughput
        }
    })


@app.post("/step")
def step(body: dict):
    task = body.get("task", TASKS[0])
    wrapped = envs.get(task)

    if wrapped is None:
        return JSONResponse({"error": "call /reset first"}, status_code=400)

    obs = last_obs.get(task)
    # Fast fallback action — no LLM calls
    action = body.get("action") or {k: 0 for k in obs.signal_phases}

    obs, reward, done, info = wrapped.step(action)
    graders[task].add_step(reward, info)
    steps[task] = steps.get(task, 0) + 1
    step_num = steps[task]
    last_obs[task] = obs

    print(f"STEP: step={step_num} reward={reward.value:.2f}", flush=True)

    if done:
        score = graders[task].grade()
        success = score >= 0.5
        print(f"END: success={str(success).lower()} steps={step_num}", flush=True)

    return JSONResponse({
        "observation": {
            "vehicle_counts": obs.vehicle_counts,
            "signal_phases": obs.signal_phases,
            "waiting_times": obs.waiting_times,
            "disruptions": obs.disruptions,
            "throughput": obs.throughput
        },
        "reward": reward.value,
        "done": done,
        "info": info
    })


def main():
    return app

if __name__ == "__main__":
    main()
