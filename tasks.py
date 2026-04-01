from env import FarmEnv, FarmAction, expected_yield, expected_profit


TASK_SCENARIOS = {
    "task1": {
        "crops": [
            {"name": "wheat", "area_hectares": 2.0, "growth_stage": 0.5},
        ],
        "resources": {"water_units": 30.0, "fertilizer_kg": 5.0, "labor_hours": 10.0},
        "weather_sequence": ["sunny", "sunny", "sunny", "sunny", "sunny", "sunny", "sunny"],
        "task_description": "Keep wheat water_level between 0.4-0.7 each day for 3 days.",
        "seed": 42,
        "enable_stress": False,
        "enable_market": False,
        "enable_forecast": False,
    },
    "task2": {
        "crops": [
            {"name": "corn", "area_hectares": 2.0, "growth_stage": 0.1, "initial_stress": 0.3},
            {"name": "tomato", "area_hectares": 1.0, "growth_stage": 0.5},
            {"name": "soybean", "area_hectares": 1.5, "growth_stage": 0.3},
        ],
        "resources": {"water_units": 25.0, "fertilizer_kg": 8.0, "labor_hours": 15.0},
        "weather_sequence": ["sunny", "drought", "drought", "cloudy", "sunny", "sunny", "sunny"],
        "task_description": "Save the stressed, dying corn crop under drought and scarcity. Manage stress recovery while keeping other crops alive.",
        "seed": 42,
        "enable_stress": True,
        "enable_market": False,
        "enable_forecast": True,
        "stress_chance": 0.2,
    },
    "task3": {
        "crops": [
            {"name": "wheat", "area_hectares": 1.5, "growth_stage": 0.1},
            {"name": "corn", "area_hectares": 2.0, "growth_stage": 0.15},
            {"name": "tomato", "area_hectares": 0.8, "growth_stage": 0.1},
            {"name": "rice", "area_hectares": 1.0, "growth_stage": 0.2},
            {"name": "soybean", "area_hectares": 1.2, "growth_stage": 0.1},
        ],
        "resources": {"water_units": 60.0, "fertilizer_kg": 25.0, "labor_hours": 40.0},
        "weather_sequence": ["sunny", "sunny", "drought", "drought", "rainy", "cloudy", "sunny"],
        "task_description": "Maximize total profit across 5 crops over a full week with dynamic market prices, weather uncertainty, and crop stress events.",
        "seed": 42,
        "enable_stress": True,
        "enable_market": True,
        "enable_forecast": True,
        "stress_chance": 0.15,
    },
}

DEFAULT_FALLBACK_ACTION = {
    "crop_index": 0,
    "water_units": 3.0,
    "fertilizer_kg": 1.0,
    "labor_hours": 2.0,
}


def _safe_call_agent(agent_fn, obs_dict: dict) -> dict:
    try:
        result = agent_fn(obs_dict)
        if not isinstance(result, dict):
            return dict(DEFAULT_FALLBACK_ACTION)
        return result
    except Exception:
        return dict(DEFAULT_FALLBACK_ACTION)


def run_task_1(agent_fn) -> float:
    """EASY: Single Crop Water Management (3 days).
    Goal: Keep wheat water_level between 0.4-0.7 each day.
    No stress, no market, no forecast uncertainty.
    """
    env = FarmEnv(scenario=TASK_SCENARIOS["task1"])
    obs = env.reset()

    day_scores = []
    for _ in range(3):
        action_dict = _safe_call_agent(agent_fn, obs.model_dump())
        action = FarmAction(**action_dict)
        obs, reward, done, info = env.step(action)

        water = obs.crops[0].water_level
        if 0.4 <= water <= 0.7:
            day_score = 1.0
        elif water < 0.4:
            day_score = max(0.0, 1.0 - (0.4 - water) * 3)
        else:
            day_score = max(0.0, 1.0 - (water - 0.7) * 3)
        day_scores.append(day_score)

    final_score = sum(day_scores) / len(day_scores)
    return round(final_score, 4)


def run_task_2(agent_fn) -> float:
    """MEDIUM: Multi-Crop Triage Under Scarcity + Stress (5 days).
    Goal: Save stressed corn under drought, manage stress recovery.
    Features: stress system, weather forecast uncertainty.
    """
    env = FarmEnv(scenario=TASK_SCENARIOS["task2"])
    obs = env.reset()

    for _ in range(5):
        action_dict = _safe_call_agent(agent_fn, obs.model_dump())
        action = FarmAction(**action_dict)
        obs, reward, done, info = env.step(action)

    final_state = env.state()
    final_corn_health = final_state.crops[0].health
    final_corn_stress = final_state.crops[0].stress_level
    total_yield = sum(expected_yield(c) for c in final_state.crops)
    max_possible = 25.0

    health_component = final_corn_health * 0.4
    stress_recovery = (1.0 - final_corn_stress) * 0.2
    yield_component = min(1.0, total_yield / max_possible) * 0.4

    score = health_component + stress_recovery + yield_component
    if final_corn_health < 0.2:
        score *= 0.3

    return round(min(1.0, max(0.0, score)), 4)


def run_task_3(agent_fn) -> float:
    """HARD: Full Week Profit Optimization (7 days).
    Goal: Maximize total profit with market prices, stress, and uncertainty.
    Features: all advanced systems enabled.
    """
    env = FarmEnv(scenario=TASK_SCENARIOS["task3"])
    obs = env.reset()

    done = False
    while not done:
        action_dict = _safe_call_agent(agent_fn, obs.model_dump())
        action = FarmAction(**action_dict)
        obs, reward, done, info = env.step(action)

    total_yield = sum(expected_yield(c) for c in env.crops)
    total_profit = sum(expected_profit(c, env.market_prices) for c in env.crops)

    yield_score = min(1.0, total_yield / 45.0)
    profit_score = min(1.0, total_profit / 55.0)
    score = yield_score * 0.4 + profit_score * 0.6

    return round(min(1.0, max(0.0, score)), 4)
