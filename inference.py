import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

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

TASKS = ["basic_intersection", "multi_intersection", "city_network"]
MAX_STEPS = 30


def parse_action(response_text: str, signal_phases: dict) -> dict:
    try:
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            action = json.loads(text[start:end])
            return {k: int(v) % 4 for k, v in action.items() if k in signal_phases}
    except Exception:
        pass
    return {k: 0 for k in signal_phases}


def run_task(task: str) -> dict:
    print(f"START: {task}", flush=True)

    env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
    wrapped = DisruptionWrapper(env)
    grader = get_grader(task)
    obs = wrapped.reset()
    grader.reset()

    rewards = []
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        signal_phases = obs.signal_phases

        if client is not None:
            try:
                prompt = (
                    f"Traffic signal control. Task: {task}. Step: {step}. "
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
                action = parse_action(response.choices[0].message.content, signal_phases)
            except Exception:
                action = {k: 0 for k in signal_phases}
        else:
            action = {k: 0 for k in signal_phases}

        obs, reward, done, info = wrapped.step(action)
        grader.add_step(reward, info)
        rewards.append(reward.value)
        print(f"STEP: step={step} reward={reward.value:.2f}", flush=True)

    wrapped.close()
    final_score = grader.grade()
    success = final_score >= 0.5
    print(f"END: success={str(success).lower()} steps={step}", flush=True)
    return {"task": task, "score": final_score, "steps": step, "success": success}


if __name__ == "__main__":
    import time
    for task in TASKS:
        run_task(task)
    while True:
        time.sleep(60)
