"""
FarmEnv Hybrid Inference Agent

MANDATORY env variables:
  API_BASE_URL  - LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    - model identifier
  HF_TOKEN      - Hugging Face token (fallback: API_KEY env var)

Architecture:
  Layer 1: Rule-based intelligence (priority scoring, weather-aware, stress-aware)
  Layer 2: LLM refinement (adjusts resource quantities)
  Fallback: Rule-based decisions always available without LLM

Rules:
- Uses OpenAI client for ALL LLM calls
- Must complete in under 20 minutes
- Must run on vcpu=2, memory=8gb
"""
import os
import re
import json
from openai import OpenAI
from tasks import run_task_1, run_task_2, run_task_3
from env import CROP_PROFILES

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "placeholder")
    return _client


DEFAULT_ACTION = {"crop_index": 0, "water_units": 3.0, "fertilizer_kg": 1.0, "labor_hours": 2.0}


def _score_crop_priority(i: int, crop: dict, market_prices: dict, day: int) -> float:
    """Score a crop by urgency: low health, high stress, high market value, growth potential."""
    health = crop.get("health", 1.0)
    stress = crop.get("stress_level", 0.0)
    growth = crop.get("growth_stage", 1.0)
    name = crop.get("name", "wheat")
    area = crop.get("area_hectares", 1.0)
    price = market_prices.get(name, 1.0)
    profile = CROP_PROFILES.get(name, {})
    base_yield = profile.get("base_yield", 1.0)

    score = 0.0
    score += (1.0 - health) * 3.0
    score += stress * 2.5
    potential = base_yield * area * price
    score += potential * 0.3
    if day <= 3:
        score += (1.0 - growth) * 1.5
    return score


def _compute_resource_allocation(best_crop: dict, resources: dict, forecast: dict, day: int) -> dict:
    """Compute resource allocation: target-based water, weather-aware, stress-aware."""
    water_left = resources.get("water_units", 0)
    fert_left = resources.get("fertilizer_kg", 0)
    labor_left = resources.get("labor_hours", 0)
    days_remaining = max(1, 7 - day)

    fert = min(fert_left / days_remaining * 1.1, fert_left)
    labor = min(labor_left / days_remaining * 1.1, labor_left)

    # Target-based water: only add what the crop actually needs
    name = best_crop.get("name", "wheat")
    profile = CROP_PROFILES.get(name, {})
    water_need = profile.get("water_need", 0.5)
    current_water = best_crop.get("water_level", 0.4)

    if current_water >= water_need:
        water = 0.0
    else:
        deficit_units = max(0.0, (water_need - current_water) * 10.0)
        safe_max = max(0.0, (water_need + 0.1 - current_water) * 10.0)
        budget_cap = water_left / max(days_remaining * 0.5, 1)
        water = min(deficit_units, safe_max, budget_cap)

    # Weather-aware adjustments
    forecast_key = f"day_{day}"
    if forecast_key in forecast:
        day_forecast = forecast[forecast_key]
        drought_prob = day_forecast.get("drought", 0)
        rain_prob = day_forecast.get("rainy", 0)
        if drought_prob > 0.4:
            water = min(water + 2.0, water_left * 0.35)
        elif rain_prob > 0.4:
            water = water * 0.5

    # Stress-aware: extra labor for stressed crops
    if best_crop.get("stressed", False):
        labor = min(labor + 2.0, labor_left * 0.35)

    water = min(max(0.0, water), water_left)

    return {
        "water_units": round(water, 1),
        "fertilizer_kg": round(max(0.0, fert), 1),
        "labor_hours": round(max(0.0, labor), 1),
    }


def rule_based_agent(observation: dict) -> dict:
    """Layer 1: Pure rule-based intelligence. No LLM needed."""
    crops = observation.get("crops", [])
    resources = observation.get("resources", {})
    market_prices = observation.get("market_prices", {})
    forecast = observation.get("forecast", {})
    day = observation.get("day", 1)

    if not crops:
        return dict(DEFAULT_ACTION)

    priorities = []
    for i, crop in enumerate(crops):
        score = _score_crop_priority(i, crop, market_prices, day)
        priorities.append((i, score, crop))

    priorities.sort(key=lambda x: x[1], reverse=True)
    best_idx = priorities[0][0]
    best_crop = priorities[0][2]

    alloc = _compute_resource_allocation(best_crop, resources, forecast, day)
    alloc["crop_index"] = best_idx
    return alloc


def llm_agent(observation: dict) -> dict:
    """Layer 2: LLM-based agent for refinement."""
    day = observation.get("day", "?")
    weather = observation.get("weather", "?")
    crops = observation.get("crops", [])
    resources = observation.get("resources", {})
    task_description = observation.get("task_description", "")
    market_prices = observation.get("market_prices", {})
    forecast = observation.get("forecast", {})

    crop_lines = []
    for i, crop in enumerate(crops):
        name = crop.get("name", "?")
        water = crop.get("water_level", 0)
        nutrients = crop.get("nutrient_level", 0)
        health_val = crop.get("health", 0)
        growth = crop.get("growth_stage", 0)
        stress = crop.get("stress_level", 0)
        stressed = crop.get("stressed", False)
        price = market_prices.get(name, 1.0)
        stress_tag = " [STRESSED]" if stressed else ""
        crop_lines.append(
            f"  [{i}] {name:8s} - water:{water:.2f} nutrients:{nutrients:.2f} "
            f"health:{health_val:.2f} growth:{growth:.2f} stress:{stress:.2f} "
            f"price:{price:.2f}{stress_tag}"
        )

    water_left = resources.get("water_units", 0)
    fert_left = resources.get("fertilizer_kg", 0)
    labor_left = resources.get("labor_hours", 0)

    forecast_text = ""
    if forecast:
        parts = []
        for k, v in sorted(forecast.items()):
            top = max(v.items(), key=lambda x: x[1])
            parts.append(f"{k}: {top[0]}({top[1]:.0%})")
        forecast_text = f"\nForecast: {', '.join(parts)}"

    crops_text = "\n".join(crop_lines)
    prompt = (
        f"You are an expert farm manager AI optimizing for profit.\n"
        f"Day: {day}/7 | Weather forecast: {weather}{forecast_text}\n"
        f"Crops:\n{crops_text}\n"
        f"Resources: water={water_left:.1f}, fertilizer={fert_left:.1f}, labor={labor_left:.1f}\n"
        f"Task: {task_description}\n\n"
        f"RULES:\n"
        f"- Prioritize: low health crops, stressed crops, high-value crops\n"
        f"- Conserve resources for remaining days\n"
        f"- Labor treats crop stress (0.05 stress reduction per labor hour)\n"
        f"- Weather may differ from forecast\n\n"
        f"Choose ONE crop. Respond ONLY with valid JSON:\n"
        f'{{"crop_index": 0, "water_units": 5.0, "fertilizer_kg": 2.0, "labor_hours": 3.0}}'
    )

    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Expert farm AI. ONLY output valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=150,
    )
    response_text = response.choices[0].message.content.strip()
    match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if match:
        action = json.loads(match.group())
        required_keys = {"crop_index", "water_units", "fertilizer_kg", "labor_hours"}
        if required_keys.issubset(action.keys()):
            return action

    return dict(DEFAULT_ACTION)


def smart_agent(observation: dict) -> dict:
    """Hybrid two-layer agent: rule-based priority + LLM refinement.

    Layer 1 always runs (no API needed).
    Layer 2 (LLM) refines quantities; crop selection trusts Layer 1.
    Falls back to pure rule-based if LLM fails.
    """
    rule_action = rule_based_agent(observation)

    try:
        llm_action = llm_agent(observation)
        blended = {
            "crop_index": rule_action["crop_index"],
            "water_units": round(
                rule_action["water_units"] * 0.6 + llm_action.get("water_units", 3.0) * 0.4, 1
            ),
            "fertilizer_kg": round(
                rule_action["fertilizer_kg"] * 0.6 + llm_action.get("fertilizer_kg", 1.0) * 0.4, 1
            ),
            "labor_hours": round(
                rule_action["labor_hours"] * 0.6 + llm_action.get("labor_hours", 2.0) * 0.4, 1
            ),
        }
        return blended
    except Exception:
        return rule_action


if __name__ == "__main__":
    print("=== FarmEnv Hybrid Agent Inference ===")
    print("Architecture: Rule-based priority scoring + LLM refinement")
    print()
    score1 = run_task_1(smart_agent)
    score2 = run_task_2(smart_agent)
    score3 = run_task_3(smart_agent)
    avg = (score1 + score2 + score3) / 3
    print(f"Task 1 (Easy)   - Water Management:      {score1:.4f}")
    print(f"Task 2 (Medium) - Triage + Stress:        {score2:.4f}")
    print(f"Task 3 (Hard)   - Profit Optimization:    {score3:.4f}")
    print(f"Average Score: {avg:.4f}")
