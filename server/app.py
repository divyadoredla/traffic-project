import os
import json
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from openai import OpenAI
from traffic_env import TrafficSignalEnv, DisruptionWrapper
from traffic_env.graders import get_grader

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

client = None
if HF_TOKEN:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        client = None

MAX_STEPS = 30
TASKS = ["basic_intersection", "multi_intersection", "city_network"]

app = FastAPI()

envs = {}
graders = {}
steps = {}


def get_action(obs, task: str) -> dict:
    signal_phases = obs.signal_phases
    if client is None:
        return {k: 0 for k in signal_phases}
    try:
        prompt = (
            f"Traffic signal control. Task: {task}. "
            f"Signal phases: {json.dumps(signal_phases)}. "
            f"Vehicle counts: {json.dumps(obs.vehicle_counts)}. "
            f"Respond ONLY with JSON like: {{\"intersection_0\": 1}}"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            action = json.loads(text[start:end])
            return {k: int(v) % 4 for k, v in action.items() if k in signal_phases}
    except Exception:
        pass
    return {k: 0 for k in signal_phases}


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

    obs = wrapped.env.last_obs if hasattr(wrapped.env, "last_obs") else wrapped.reset()
    action = body.get("action") or get_action(obs, task)

    obs, reward, done, info = wrapped.step(action)
    graders[task].add_step(reward, info)
    steps[task] = steps.get(task, 0) + 1
    step_num = steps[task]

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
