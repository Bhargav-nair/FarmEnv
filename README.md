---
title: FarmEnv
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# FarmEnv - AI Farm Resource Allocation Environment

🚀 Built for OpenEnv Hackathon — Focused on real-world AI decision systems under uncertainty

## Overview

This environment simulates real-world Agri-AI decision systems used in precision farming and resource optimization.

FarmEnv is an OpenEnv-compliant simulation environment where an AI agent manages a farm over 7 days, making daily decisions about how to allocate limited water, fertilizer, and labor across multiple crops to maximize total profit.

Agriculture is one of humanity's most critical challenges. With growing populations, climate volatility, and shrinking arable land, the need for precision farming and intelligent resource allocation has never been greater. FarmEnv distills these real-world complexities into a tractable decision-making environment:

- **Food Security**: Optimizing crop yields with constrained resources mirrors real agricultural planning under scarcity.
- **Precision Farming**: Crop-specific needs, weather uncertainty, market dynamics, and stress events model the complexity of modern AgTech.
- **Climate Adaptation**: Probabilistic weather forecasts and surprise weather events force the agent to plan under genuine uncertainty.
- **Economic Decision-Making**: Dynamic market prices transform farming into profit optimization, not just yield maximization.
- **Crisis Management**: Random crop stress/disease events require triage and recovery, reflecting real agricultural emergencies.


## What Makes FarmEnv Different

| Feature | Typical Env | FarmEnv |
|---------|-------------|---------|
| Weather | Fixed sequence | Probabilistic forecasts + hidden actual weather |
| Rewards | Simple health-based | 4-component: health + efficiency + balance + profit |
| Market | None | Dynamic daily prices per crop |
| Stress | None | Random disease/pest events requiring triage |
| Agent | Basic LLM | Hybrid: rule-based priority + LLM refinement |
| Logging | Minimal | Full decision context with risk assessment |

## Advanced Features

### Weather Uncertainty System
The agent receives probabilistic weather forecasts, not deterministic weather. Actual weather is sampled from the distribution and may differ from the forecast. This forces planning under uncertainty.

```json
"forecast": {
  "day_1": {"sunny": 0.85, "cloudy": 0.05, "rainy": 0.05, "drought": 0.05},
  "day_2": {"sunny": 0.05, "cloudy": 0.05, "rainy": 0.05, "drought": 0.85},
  "day_3": {"rainy": 0.55, "sunny": 0.15, "cloudy": 0.15, "drought": 0.15}
}
```

### Crop Stress / Disease System
Each day, crops have a 10-20% chance of developing stress (disease, pests, etc.). Stress reduces health and requires labor investment to treat. Ignoring stressed crops incurs reward penalties.

- `stress_level` (0.0-1.0): Current stress intensity
- `stressed` (bool): Whether crop is under active stress
- Labor reduces stress: 0.05 stress reduction per labor hour
- Penalty: -0.1 reward if any crop has stress > 0.5

### Dynamic Market Prices
Each crop has a base market price that fluctuates daily. Final profit = yield x market_price. This transforms the environment from pure farming into economic decision-making.

| Crop | Base Price | Range |
|------|-----------|-------|
| Wheat | 1.20 | 0.90-1.50 |
| Corn | 1.50 | 1.20-1.80 |
| Tomato | 2.80 | 2.50-3.10 |
| Rice | 1.80 | 1.50-2.10 |
| Soybean | 1.00 | 0.70-1.30 |

### Advanced Reward Function

FarmEnv uses a multi-objective reward system:

reward = (
  health_score * 0.3 +
  efficiency_score * 0.2 +
  balance_score * 0.2 +
  profit_score * 0.3
) + penalties

- Health: Crop health after action
- Efficiency: Output per resource used
- Balance: Avoid neglecting crops
- Profit: Economic return from yield

Penalties:
- Overwatering (-0.1)
- Stress neglect (-0.1)
### Decision Logging
Every step generates interpretable context:
```json
"decision_context": {
  "priority_crop": "corn",
  "risk_level": "high",
  "market_opportunity": "tomato",
  "resource_pressure": "tight",
  "weather_surprise": true,
  "stress_alert": true
}
```

### Hybrid Agent Architecture
The inference agent uses a two-layer design:
- **Layer 1 (Rule-based)**: Scores crops by health urgency, stress level, market value, and growth potential. Allocates resources using target-based water management and weather-aware adjustments.
- **Layer 2 (LLM refinement)**: Optionally refines resource quantities using an LLM. Crop selection trusts the rule-based layer.
- **Fallback**: Rule-based decisions always available without LLM API access.

## Environment Description

### Crops

FarmEnv supports five crop types, each with unique water needs, nutrient requirements, and base yield potential:

| Crop    | Water Need | Nutrient Need | Base Yield |
|---------|-----------|---------------|------------|
| Wheat   | 0.5       | 0.4           | 3.0        |
| Corn    | 0.7       | 0.6           | 5.0        |
| Tomato  | 0.8       | 0.7           | 7.0        |
| Rice    | 0.9       | 0.5           | 4.5        |
| Soybean | 0.5       | 0.3           | 2.5        |

Each crop tracks:
- **water_level** (0.0-1.0): Current hydration. Too low or too high relative to the crop's need reduces health.
- **nutrient_level** (0.0-1.0): Soil nutrient concentration. Closer to the crop's need is better.
- **growth_stage** (0.0-1.0): Maturity progress. Increases with labor investment and passive daily growth.
- **health** (0.0-1.0): Computed from how well water and nutrient levels match the crop's ideal profile.
- **area_hectares**: Plot size, multiplies final yield.

### Weather System

Weather changes daily according to a predefined sequence. Each condition affects water dynamics:

| Weather  | Water Bonus |
|----------|------------|
| Sunny    | +0.0       |
| Cloudy   | +0.1       |
| Rainy    | +0.3       |
| Drought  | -0.1       |

The weather bonus is applied to the targeted crop's water level alongside manual watering each step.

### Resource Mechanics

The agent has three finite resource pools shared across all crops:

- **water_units**: Used for irrigation. Each unit adds 0.1 to the targeted crop's water level.
- **fertilizer_kg**: Used for nutrition. Each kg adds 0.125 to the targeted crop's nutrient level.
- **labor_hours**: Used for cultivation. Each hour adds 0.025 to the targeted crop's growth stage (plus 0.02 passive daily growth).

Resources are consumed per action and never replenish. If the agent requests more than available, usage is silently clamped to the remaining amount.

### Episode Structure

1. **Reset**: All crops initialize with `water_level=0.4` and `nutrient_level=0.4`. Growth stages and areas come from the scenario.
2. **Steps**: Each step the agent targets ONE crop, allocating water, fertilizer, and labor from the shared pool.
3. **Completion**: The episode ends after 6 steps (spanning day 1 through day 7). On the final step, a yield-based bonus is added to the reward.

### Health Calculation

```
water_score  = 1.0 - |water_level - water_need|
nutrient_score = 1.0 - |nutrient_level - nutrient_need|
health = clamp(water_score * 0.6 + nutrient_score * 0.4, 0.0, 1.0)
```

### Yield Calculation

```
yield_per_crop = base_yield * health * growth_stage * area_hectares
```

## Observation Space

| Field            | Type                    | Range         | Description                                      |
|------------------|-------------------------|---------------|--------------------------------------------------|
| day              | int                     | 1-7           | Current simulation day                           |
| weather          | WeatherCondition (enum) | sunny/cloudy/rainy/drought | Current weather affecting water dynamics |
| crops            | list[CropState]         | Variable      | State of each crop (water, nutrients, health, growth, area) |
| resources        | dict                    | Non-negative  | Remaining `water_units`, `fertilizer_kg`, `labor_hours` |
| cumulative_score | float                   | Any           | Running total of all rewards earned so far       |
| task_description | str                     | -             | Human-readable description of the current goal   |
| forecast         | dict                    | -             | Probabilistic weather predictions for upcoming days |
| market_prices    | dict                    | > 0           | Current market price per crop type               |

### CropState Fields

| Field          | Type      | Range     | Description                          |
|----------------|-----------|-----------|--------------------------------------|
| name           | CropType  | enum      | Crop identifier (wheat/corn/etc.)    |
| water_level    | float     | 0.0-1.0   | Current hydration level              |
| nutrient_level | float     | 0.0-1.0   | Current soil nutrient concentration  |
| growth_stage   | float     | 0.0-1.0   | Maturity progress                    |
| health         | float     | 0.0-1.0   | Computed crop health score           |
| area_hectares  | float     | > 0       | Plot size in hectares                |
| stress_level   | float     | 0.0-1.0   | Current stress from disease/pests    |
| stressed       | bool      | T/F       | Whether crop is under active stress  |

## Action Space

| Field         | Type  | Range         | Description                                |
|---------------|-------|---------------|--------------------------------------------|
| crop_index    | int   | 0 to N-1      | Index of the crop to target this step      |
| water_units   | float | >= 0          | Water to allocate (clamped to available)   |
| fertilizer_kg | float | >= 0          | Fertilizer to allocate (clamped to available) |
| labor_hours   | float | >= 0          | Labor to allocate (clamped to available)   |

**Notes:**
- Each step targets exactly ONE crop.
- If `crop_index` is out of range, a penalty of -0.1 is applied and the episode continues.
- Resource requests exceeding the available amount are silently clamped.

## Reward Function

The reward for each step has multiple components:

### Step Reward

```
step_reward = (crop_health * 0.4) + efficiency_bonus + waste_penalty
```

Where:
- **crop_health * 0.4**: Proportional to the targeted crop's health after applying resources. Rewards keeping crops healthy.
- **efficiency_bonus**: `min(0.2, (crop_health / max(water_used + fertilizer_used, 1.0)) * 2.0)`. Rewards achieving high health with minimal resource expenditure.
- **waste_penalty**: `-0.1` if the crop's water level exceeds `water_need + 0.2` (over-watering). Otherwise `0.0`.

### Final Bonus (on episode completion)

```
final_yield_bonus = min(1.0, total_yield / 50.0)
```

Where `total_yield = sum(base_yield * health * growth_stage * area for each crop)`. Added to the last step's reward.

### Invalid Action Penalty

If `crop_index` is out of range: `reward = -0.1`, episode continues.

### Reward Range

Overall reward per step typically falls in `[-1.0, 2.0]`.

## Tasks

### Task 1 - Single Crop Water Management (Easy)

**Scenario:**
- 1 wheat plot (2 hectares, growth_stage=0.5)
- Resources: water=30, fertilizer=5, labor=10
- Weather: sunny every day
- Duration: 3 steps

**Goal:** Keep wheat's `water_level` between 0.4 and 0.7 after each step.

**What a good agent does:** Applies moderate water each day (around 1-3 units) to maintain the optimal range without over-watering. The sunny weather provides no water bonus, so all hydration comes from manual irrigation.

**Grading:**
```
For each of 3 days:
  water = crop.water_level after step
  if 0.4 <= water <= 0.7:    day_score = 1.0
  elif water < 0.4:           day_score = max(0.0, 1.0 - (0.4 - water) * 3)
  else:                       day_score = max(0.0, 1.0 - (water - 0.7) * 3)
final_score = average of 3 day_scores  (range: 0.0-1.0)
```

### Task 2 - Multi-Crop Triage Under Scarcity (Medium)

**Scenario:**
- Corn (2ha, growth=0.1) - critical, nearly dying
- Tomato (1ha, growth=0.5) - healthy
- Soybean (1.5ha, growth=0.3) - moderate
- Resources: water=25, fertilizer=8, labor=15 (intentionally scarce)
- Weather: sunny, drought, drought, cloudy, sunny
- Duration: 5 steps

**Goal:** Recognize that corn is dying and prioritize it while keeping other crops alive.

**What a good agent does:** Focuses water and fertilizer on corn (especially during drought days), allocates some labor to boost corn's growth stage, and distributes remaining resources to maintain tomato and soybean.

**Grading:**
```
final_corn_health = env.state().crops[0].health after 5 steps
total_yield = sum of expected_yield for all crops
max_possible = 25.0

score = (final_corn_health * 0.5) + (min(1.0, total_yield / max_possible) * 0.5)
if final_corn_health < 0.2:  score *= 0.3  (heavy penalty if corn dies)
return round(score, 4)
```

### Task 3 - Week-Long Yield Optimization (Hard)

**Scenario:**
- Wheat (1.5ha, growth=0.1), Corn (2ha, growth=0.15), Tomato (0.8ha, growth=0.1), Rice (1ha, growth=0.2), Soybean (1.2ha, growth=0.1)
- Resources: water=60, fertilizer=25, labor=40
- Weather: sunny, sunny, drought, drought, rainy, cloudy, sunny
- Duration: Full episode (runs to completion)

**Goal:** Maximize total yield across all 5 crops.

**What a good agent does:** Distributes resources across all crops based on their yield potential and current needs. Prioritizes high-base-yield crops (tomato, corn) while accounting for weather effects. Uses rainy days to focus labor/fertilizer instead of water. Plans resource expenditure across the full week.

**Grading:**
```
total_yield = sum(base_yield * health * growth_stage * area for each crop)
score = min(1.0, total_yield / 45.0)
return round(score, 4)
```

## Setup & Local Usage

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
cd farm-env
pip install -r requirements.txt
```

### Running the Server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

The server starts with an auto-reset, so `/state` works immediately.

### API Endpoints

| Method | Path    | Description                           |
|--------|---------|---------------------------------------|
| POST   | /reset  | Reset env (optional: `{"task": "task1"}`) |
| POST   | /step   | Take an action                        |
| GET    | /state  | Get current observation               |
| GET    | /tasks  | List available tasks                  |
| GET    | /health | Health check                          |

### Example API Calls

```bash
# Health check
curl http://localhost:7860/health

# Reset with default scenario
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{}'

# Reset with specific task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task": "task2"}'

# Take a step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" \
  -d '{"crop_index": 0, "water_units": 5.0, "fertilizer_kg": 2.0, "labor_hours": 3.0}'

# Get current state
curl http://localhost:7860/state

# List tasks
curl http://localhost:7860/tasks
```

## Docker Usage

### Build

```bash
docker build -t farm-env .
```

### Run

```bash
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=your_model \
  farm-env
```

## Running Inference

The baseline agent uses an LLM to make farming decisions. Configure environment variables and run:

```bash
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Output Format

```
=== FarmEnv Baseline Inference ===
Task 1 (Easy)   - Water Management:      0.XXXX
Task 2 (Medium) - Triage Under Scarcity:  0.XXXX
Task 3 (Hard)   - Full Optimization:      0.XXXX
Average Score: 0.XXXX
```

## Baseline Scores

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) | Average |
|-------|--------------|----------------|---------------|---------|
| Dumb (constant action) | 0.4000 | 0.5251 | 0.0892 | 0.3381 |
| Rule-based (no LLM)    | 1.0000 | 0.5720 | 0.2515 | 0.6078 |
| Hybrid (rule + LLM)    | TBD    | TBD    | TBD    | TBD    |

## Environment Variables

| Variable      | Required | Default                                  | Description                        |
|---------------|----------|------------------------------------------|------------------------------------|
| API_BASE_URL  | No       | https://router.huggingface.co/v1         | LLM API endpoint                   |
| MODEL_NAME    | Yes*     | -                                        | Model identifier for LLM calls     |
| HF_TOKEN      | Yes*     | -                                        | Hugging Face API token             |
| API_KEY       | No       | -                                        | Fallback API key (if HF_TOKEN not set) |

*Required only for running `inference.py`. The server itself runs without LLM credentials.

## File Structure

```
farm-env/
├── env.py           # Core environment: Pydantic models, crop physics, FarmEnv class
├── tasks.py         # Task definitions, scenarios, and grading functions
├── server/
│   └── app.py       # FastAPI server with REST endpoints
├── inference.py     # Baseline LLM agent
├── openenv.yaml     # OpenEnv specification
├── Dockerfile       # Container configuration for HF Spaces
├── requirements.txt # Python dependencies
└── README.md        # This file
```

## License

This project is part of the OpenEnv hackathon submission.
