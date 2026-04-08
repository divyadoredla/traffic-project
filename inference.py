import os
import json
from openai import OpenAI
from traffic_env import TrafficSignalEnv, DisruptionWrapper
from traffic_env.graders import get_grader

# Environment variables - no hardcoded defaults
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["basic_intersection", "multi_intersection", "city_network"]
MAX_STEPS = 30  # Increased from 50 for better performance


def obs_to_prompt(obs, task: str, step: int) -> str:
    """Convert observation to LLM prompt"""
    vehicle_counts = obs.vehicle_counts
    signal_phases = obs.signal_phases
    waiting_times = obs.waiting_times
    disruptions = obs.disruptions
    throughput = obs.throughput

    active_disruptions = [lane for lane, blocked in disruptions.items() if blocked]
    num_intersections = len(signal_phases)

    # Calculate priority scores for better decision making
    lane_priorities = {}
    for lane_id in vehicle_counts.keys():
        if lane_id in waiting_times and lane_id in disruptions:
            # Higher priority = more vehicles + longer wait + disruption penalty
            priority = vehicle_counts[lane_id] * (1 + waiting_times[lane_id] / 100)
            if disruptions[lane_id]:
                priority *= 0.1  # Heavily deprioritize blocked lanes
            lane_priorities[lane_id] = round(priority, 2)
    
    # Get top priority lanes
    top_lanes = sorted(lane_priorities.items(), key=lambda x: x[1], reverse=True)[:6]
    
    prompt = f"""You are an expert traffic signal controller optimizing for minimal waiting times and maximum throughput.

CURRENT SITUATION:
- Task: {task} | Step: {step}/50
- Throughput: {throughput:.2f} (Target: >0.75)
- Active Disruptions: {len(active_disruptions)} lanes blocked

CRITICAL LANES (highest priority):
{json.dumps(dict(top_lanes), indent=2)}

CURRENT SIGNAL PHASES:
{json.dumps(signal_phases, indent=2)}

STRATEGY:
1. Prioritize lanes with highest priority scores (vehicles × waiting time)
2. Alternate phases every 3-5 steps to prevent starvation
3. AVOID blocked lanes - they have 0.1x priority
4. Balance north-south (phases 0,2) vs east-west (phases 1,3)
5. If throughput < 0.6, switch phases more frequently

BLOCKED LANES: {active_disruptions if active_disruptions else "None - all clear!"}

Respond with ONLY a JSON object:
{{"intersection_0": 1, "intersection_1": 0, "intersection_2": 2}}

Available intersections: {list(signal_phases.keys())}
"""
    return prompt


def parse_action(response_text: str, signal_phases: dict) -> dict:
    """Parse LLM response into action dict"""
    try:
        # Extract JSON from response
        text = response_text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            action = json.loads(text[start:end])
            # Validate and clamp phases to 0-3
            return {k: int(v) % 4 for k, v in action.items() if k in signal_phases}
    except Exception:
        pass
    # Fallback: keep current phases
    return {k: v % 4 for k, v in signal_phases.items()}


def run_task(task: str) -> dict:
    """Run a single task episode and return results"""
    env = TrafficSignalEnv(task=task, max_steps=MAX_STEPS)
    wrapped_env = DisruptionWrapper(env)
    grader = get_grader(task)

    obs = wrapped_env.reset()
    grader.reset()

    rewards = []
    last_error = None
    done = False
    step = 0

    print(f"[START] task={task} env=traffic-signal-control model={MODEL_NAME}")

    while not done and step < MAX_STEPS:
        step += 1
        prompt = obs_to_prompt(obs, task, step)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=300
            )
            action_text = response.choices[0].message.content
            action = parse_action(action_text, obs.signal_phases)
            last_error = None
        except Exception as e:
            # Fallback to phase 0 for all intersections on API error
            action = {k: 0 for k in obs.signal_phases}
            last_error = str(e)[:100]

        obs, reward, done, info = wrapped_env.step(action)
        grader.add_step(reward, info)
        rewards.append(reward.value)

        error_str = last_error if last_error else "null"
        action_str = json.dumps(action)
        print(
            f"[STEP]  step={step} action={action_str} "
            f"reward={reward.value:.2f} done={str(done).lower()} error={error_str}"
        )

    wrapped_env.close()

    final_score = grader.grade()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success = final_score >= 0.5

    print(f"[END]   success={str(success).lower()} steps={step} rewards={rewards_str}")

    return {"task": task, "score": final_score, "steps": step, "success": success}


if __name__ == "__main__":
    results = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\nBaseline Results:")
    for r in results:
        print(f"  {r['task']}: score={r['score']:.3f} success={r['success']}")
    print(f"  Average score: {avg_score:.3f}")