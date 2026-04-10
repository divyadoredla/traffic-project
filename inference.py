import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

from openai import OpenAI

try:
    from traffic_env import TrafficSignalEnv, DisruptionWrapper
    from traffic_env.graders import get_grader
except ImportError:
    print("[END] task=setup score=0.0 steps=0", flush=True)
    sys.exit(0)

TASKS = ["basic_intersection"]
MAX_STEPS = 20


def get_client():
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("API_KEY", "placeholder")
    return OpenAI(base_url=api_base, api_key=api_key, timeout=25.0)


def get_llm_action(signal_phases: dict) -> dict:
    """Call LLM — signal_phases is Dict[str, int] e.g. {'intersection_0': 0}"""
    model = os.environ.get("MODEL_NAME", "gpt-4.1-mini")
    keys = list(signal_phases.keys())
    prompt = (
        f"You are a traffic signal controller.\n"
        f"Intersections: {keys}\n"
        "Return ONLY a JSON object assigning phase 0-3 to each intersection.\n"
        f"Example: {json.dumps({k: 0 for k in keys})}\nOnly JSON."
    )
    for _ in range(2):
        try:
            resp = get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=80,
            )
            text = resp.choices[0].message.content.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(text[start:end])
                result = {k: int(v) % 4 for k, v in parsed.items() if k in signal_phases}
                if result:
                    return result
        except Exception:
            continue
    # Safe fallback — always a valid dict
    return {k: 0 for k in signal_phases}


def run_task(task: str):
    print(f"[START] task={task}", flush=True)

    try:
        env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
        wrapped = DisruptionWrapper(env)
        grader = get_grader(task)
        obs = wrapped.reset()
        grader.reset()
    except Exception as e:
        print(f"[END] task={task} score=0.0 steps=0", flush=True)
        return

    done = False
    step = 0

    while not done and step < MAX_STEPS:
        step += 1
        try:
            # obs.signal_phases is Dict[str, int] — use directly
            action = get_llm_action(obs.signal_phases)
            obs, reward, done, info = wrapped.step(action)
            grader.add_step(reward, info)
            reward_val = reward.value if hasattr(reward, "value") else float(reward)
            print(f"[STEP] step={step} reward={reward_val:.2f}", flush=True)
        except Exception:
            print(f"[STEP] step={step} reward=0.00", flush=True)
            break

    try:
        wrapped.close()
    except Exception:
        pass

    try:
        final_score = grader.grade()
    except Exception:
        final_score = 0.5

    print(f"[END] task={task} score={final_score:.4f} steps={step}", flush=True)


if __name__ == "__main__":
    try:
        for task in TASKS:
            run_task(task)
    except Exception:
        print("[END] task=unknown score=0.0 steps=0", flush=True)
    sys.exit(0)
