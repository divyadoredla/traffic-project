import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI
from traffic_env import TrafficSignalEnv, DisruptionWrapper
from traffic_env.graders import get_grader

TASKS = ["basic_intersection"]
MAX_STEPS = 20


def get_client():
    """Read env vars fresh every call — evaluator injects them at runtime."""
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("HF_TOKEN", "") or os.environ.get("API_KEY", "")
    return OpenAI(
        base_url=api_base,
        api_key=api_key if api_key else "placeholder",
        timeout=25.0
    )


def get_llm_action(obs, task: str) -> dict:
    """Call LLM through hackathon proxy — mandatory every step."""
    signal_phases = obs.signal_phases
    client = get_client()
    model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    prompt = (
        f"You are a traffic signal controller. Task: {task}.\n"
        f"Signal phases available: {list(signal_phases.keys())}\n"
        f"Vehicle counts: {json.dumps(obs.vehicle_counts)}\n"
        "Respond ONLY with a JSON object like: {\"intersection_0\": 1}\n"
        "No explanation. Only JSON."
    )

    # Primary attempt
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
        )
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
            return {k: int(v) % 4 for k, v in parsed.items() if k in signal_phases}
    except Exception:
        pass

    # Retry with simpler prompt
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Pick signal phases for {list(signal_phases.keys())}. Reply JSON only."}],
            max_tokens=50,
        )
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(text[start:end])
            return {k: int(v) % 4 for k, v in parsed.items() if k in signal_phases}
    except Exception:
        pass

    # Fallback only if both attempts fail
    return {k: 0 for k in signal_phases}


def run_task(task: str):
    print(f"[START] task={task}", flush=True)

    env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
    wrapped_env = DisruptionWrapper(env)
    grader = get_grader(task)
    obs = wrapped_env.reset()
    grader.reset()

    step = 0
    done = False

    while not done and step < MAX_STEPS:
        step += 1
        action = get_llm_action(obs, task)
        obs, reward, done, info = wrapped_env.step(action)
        grader.add_step(reward, info)
        reward_val = reward.value if hasattr(reward, "value") else float(reward)
        print(f"[STEP] step={step} reward={reward_val:.2f}", flush=True)
        if done:
            break

    wrapped_env.close()
    final_score = grader.grade()
    print(f"[END] task={task} score={final_score:.4f} steps={step}", flush=True)


if __name__ == "__main__":
    try:
        for task in TASKS:
            run_task(task)
        sys.exit(0)
    except Exception as e:
        print(f"[END] task=basic_intersection score=0.0 steps=0", flush=True)
        sys.exit(1)
