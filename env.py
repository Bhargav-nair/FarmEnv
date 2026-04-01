import random
from enum import Enum
from pydantic import BaseModel


class CropType(str, Enum):
    wheat = "wheat"
    corn = "corn"
    tomato = "tomato"
    rice = "rice"
    soybean = "soybean"


class WeatherCondition(str, Enum):
    sunny = "sunny"
    cloudy = "cloudy"
    rainy = "rainy"
    drought = "drought"


ALL_WEATHER = ["sunny", "cloudy", "rainy", "drought"]


class CropState(BaseModel):
    name: CropType
    water_level: float
    nutrient_level: float
    growth_stage: float
    health: float
    area_hectares: float
    stress_level: float = 0.0
    stressed: bool = False


class FarmObservation(BaseModel):
    day: int
    weather: WeatherCondition
    crops: list[CropState]
    resources: dict
    cumulative_score: float
    task_description: str
    forecast: dict = {}
    market_prices: dict = {}


class FarmAction(BaseModel):
    crop_index: int
    water_units: float
    fertilizer_kg: float
    labor_hours: float


class FarmReward(BaseModel):
    value: float
    breakdown: dict
    done: bool
    info: dict


CROP_PROFILES = {
    "wheat":   {"water_need": 0.5, "nutrient_need": 0.4, "base_yield": 3.0},
    "corn":    {"water_need": 0.7, "nutrient_need": 0.6, "base_yield": 5.0},
    "tomato":  {"water_need": 0.8, "nutrient_need": 0.7, "base_yield": 7.0},
    "rice":    {"water_need": 0.9, "nutrient_need": 0.5, "base_yield": 4.5},
    "soybean": {"water_need": 0.5, "nutrient_need": 0.3, "base_yield": 2.5},
}

WEATHER_WATER_BONUS = {
    "sunny": 0.0,
    "cloudy": 0.1,
    "rainy": 0.3,
    "drought": -0.1,
}

BASE_MARKET_PRICES = {
    "wheat": 1.2,
    "corn": 1.5,
    "tomato": 2.8,
    "rice": 1.8,
    "soybean": 1.0,
}

STRESS_CHANCE = 0.15
STRESS_HEALTH_PENALTY = 0.15
STRESS_LABOR_REDUCTION = 0.05


def calculate_health(crop: CropState) -> float:
    profile = CROP_PROFILES[crop.name.value]
    water_score = 1.0 - abs(crop.water_level - profile["water_need"])
    nutrient_score = 1.0 - abs(crop.nutrient_level - profile["nutrient_need"])
    base_health = (water_score * 0.6) + (nutrient_score * 0.4)
    stress_penalty = crop.stress_level * STRESS_HEALTH_PENALTY
    return max(0.0, min(1.0, base_health - stress_penalty))


def expected_yield(crop: CropState) -> float:
    profile = CROP_PROFILES[crop.name.value]
    return profile["base_yield"] * crop.health * crop.growth_stage * crop.area_hectares


def expected_profit(crop: CropState, market_prices: dict) -> float:
    price = market_prices.get(crop.name.value, 1.0)
    return expected_yield(crop) * price


class FarmEnv:
    def __init__(self, scenario=None):
        self.scenario = scenario or self._default_scenario()
        self._initialized = False
        self.crops: list[CropState] = []
        self.resources: dict = {}
        self.day: int = 1
        self.weather_index: int = 0
        self.cumulative_score: float = 0.0
        self.weather_sequence: list[str] = self.scenario["weather_sequence"]
        self.task_description: str = self.scenario["task_description"]

        seed = self.scenario.get("seed", 42)
        self._rng = random.Random(seed)
        self._enable_stress = self.scenario.get("enable_stress", True)
        self._enable_market = self.scenario.get("enable_market", True)
        self._enable_forecast = self.scenario.get("enable_forecast", True)
        self._stress_chance = self.scenario.get("stress_chance", STRESS_CHANCE)

        self.market_prices: dict = {}
        self._actual_weather: str = ""

    @staticmethod
    def _default_scenario() -> dict:
        return {
            "crops": [
                {"name": "wheat", "area_hectares": 2.0, "growth_stage": 0.3},
                {"name": "corn", "area_hectares": 1.5, "growth_stage": 0.2},
                {"name": "tomato", "area_hectares": 1.0, "growth_stage": 0.4},
            ],
            "resources": {
                "water_units": 50.0,
                "fertilizer_kg": 20.0,
                "labor_hours": 30.0,
            },
            "weather_sequence": [
                "sunny", "sunny", "cloudy", "rainy", "drought", "sunny", "cloudy",
            ],
            "task_description": "Maximize total profit under weather uncertainty, stress events, and dynamic market conditions.",
            "seed": 42,
            "enable_stress": True,
            "enable_market": True,
            "enable_forecast": True,
        }

    def _generate_forecast(self, from_day: int, lookahead: int = 3) -> dict:
        if not self._enable_forecast:
            return {}
        forecast = {}
        for offset in range(lookahead):
            target_day = from_day + offset
            if target_day > 7:
                break
            base_idx = min(target_day - 1, len(self.weather_sequence) - 1)
            base_weather = self.weather_sequence[base_idx]
            accuracy = max(0.4, 0.85 - offset * 0.15)
            probs = {}
            remaining = 1.0 - accuracy
            for w in ALL_WEATHER:
                if w == base_weather:
                    probs[w] = round(accuracy, 2)
                else:
                    probs[w] = round(remaining / 3, 2)
            forecast[f"day_{target_day}"] = probs
        return forecast

    def _sample_actual_weather(self) -> str:
        w_idx = min(self.weather_index, len(self.weather_sequence) - 1)
        base_weather = self.weather_sequence[w_idx]
        if not self._enable_forecast:
            return base_weather
        if self._rng.random() < 0.75:
            return base_weather
        others = [w for w in ALL_WEATHER if w != base_weather]
        return self._rng.choice(others)

    def _generate_market_prices(self) -> dict:
        if not self._enable_market:
            return {name: 1.0 for name in BASE_MARKET_PRICES}
        prices = {}
        for crop_name, base_price in BASE_MARKET_PRICES.items():
            fluctuation = self._rng.uniform(-0.3, 0.3)
            prices[crop_name] = round(max(0.3, base_price + fluctuation), 2)
        return prices

    def _apply_stress_events(self):
        if not self._enable_stress:
            return
        for crop in self.crops:
            if self._rng.random() < self._stress_chance:
                stress_amount = self._rng.uniform(0.1, 0.4)
                crop.stress_level = min(1.0, crop.stress_level + stress_amount)
                crop.stressed = True

    def reset(self) -> FarmObservation:
        self.day = 1
        self.weather_index = 0
        self.cumulative_score = 0.0
        self.resources = dict(self.scenario["resources"])
        self._rng = random.Random(self.scenario.get("seed", 42))

        self.market_prices = self._generate_market_prices()

        self.crops = []
        for crop_data in self.scenario["crops"]:
            initial_stress = crop_data.get("initial_stress", 0.0)
            crop = CropState(
                name=CropType(crop_data["name"]),
                water_level=0.4,
                nutrient_level=0.4,
                growth_stage=crop_data["growth_stage"],
                health=0.0,
                area_hectares=crop_data["area_hectares"],
                stress_level=initial_stress,
                stressed=initial_stress > 0,
            )
            crop.health = calculate_health(crop)
            self.crops.append(crop)

        self._actual_weather = self._sample_actual_weather()
        self._initialized = True
        return self._get_observation()

    def step(self, action: FarmAction) -> tuple:
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        breakdown = {}
        info = {}

        # Validate crop_index
        if action.crop_index < 0 or action.crop_index >= len(self.crops):
            self.day += 1
            self.weather_index += 1
            done = self.day >= 7

            step_reward = -0.1
            breakdown["invalid_crop_penalty"] = -0.1

            if done:
                total_yield = sum(expected_yield(c) for c in self.crops)
                total_profit = sum(expected_profit(c, self.market_prices) for c in self.crops)
                final_bonus = min(1.0, total_profit / 80.0)
                step_reward += final_bonus
                breakdown["final_yield_bonus"] = round(final_bonus, 4)
                info["total_yield"] = round(total_yield, 4)
                info["total_profit"] = round(total_profit, 4)

            self.cumulative_score += step_reward
            self._apply_stress_events()
            self.market_prices = self._generate_market_prices()
            self._actual_weather = self._sample_actual_weather()

            reward = FarmReward(
                value=round(step_reward, 4),
                breakdown=breakdown,
                done=done,
                info={"error": "Invalid crop_index"},
            )
            return self._get_observation(), reward, done, info

        # Use ACTUAL weather (may differ from forecast)
        actual_weather = self._actual_weather
        weather_bonus = WEATHER_WATER_BONUS.get(actual_weather, 0.0)

        # Clamp resource usage to available
        water_used = max(0.0, min(action.water_units, self.resources["water_units"]))
        fertilizer_used = max(0.0, min(action.fertilizer_kg, self.resources["fertilizer_kg"]))
        labor_used = max(0.0, min(action.labor_hours, self.resources["labor_hours"]))

        # Deduct resources
        self.resources["water_units"] -= water_used
        self.resources["fertilizer_kg"] -= fertilizer_used
        self.resources["labor_hours"] -= labor_used

        # Apply to targeted crop
        crop = self.crops[action.crop_index]
        crop.water_level = max(0.0, min(1.0, crop.water_level + (water_used / 10.0) + weather_bonus))
        crop.nutrient_level = max(0.0, min(1.0, crop.nutrient_level + fertilizer_used / 8.0))
        crop.growth_stage = max(0.0, min(1.0, crop.growth_stage + labor_used / 40.0 + 0.02))

        # Labor reduces stress on targeted crop
        if crop.stressed and labor_used > 0:
            stress_reduction = labor_used * STRESS_LABOR_REDUCTION
            crop.stress_level = max(0.0, crop.stress_level - stress_reduction)
            if crop.stress_level < 0.05:
                crop.stress_level = 0.0
                crop.stressed = False

        # Recalculate health
        crop.health = calculate_health(crop)

        # === ADVANCED MULTI-COMPONENT REWARD ===
        profile = CROP_PROFILES[crop.name.value]

        # Health score (0-1): targeted crop health
        health_score = crop.health

        # Efficiency score (0-1): health per resource unit
        total_res_used = water_used + fertilizer_used + labor_used
        efficiency_score = min(1.0, crop.health / max(total_res_used, 1.0) * 3.0)

        # Balance score (0-1): health uniformity across all crops
        avg_health = sum(c.health for c in self.crops) / len(self.crops)
        health_var = sum((c.health - avg_health) ** 2 for c in self.crops) / len(self.crops)
        balance_score = max(0.0, 1.0 - health_var * 4)

        # Profit score (0-1): estimated total profit
        total_profit = sum(expected_profit(c, self.market_prices) for c in self.crops)
        profit_score = min(1.0, total_profit / 80.0)

        # Penalties
        waste_penalty = -0.1 if crop.water_level > profile["water_need"] + 0.2 else 0.0
        max_stress = max(c.stress_level for c in self.crops)
        stress_neglect_penalty = -0.1 if max_stress > 0.5 else 0.0

        step_reward = (
            health_score * 0.3
            + efficiency_score * 0.2
            + balance_score * 0.2
            + profit_score * 0.3
            + waste_penalty
            + stress_neglect_penalty
        )

        breakdown["health_score"] = round(health_score * 0.3, 4)
        breakdown["efficiency_score"] = round(efficiency_score * 0.2, 4)
        breakdown["balance_score"] = round(balance_score * 0.2, 4)
        breakdown["profit_score"] = round(profit_score * 0.3, 4)
        breakdown["waste_penalty"] = round(waste_penalty, 4)
        breakdown["stress_neglect_penalty"] = round(stress_neglect_penalty, 4)
        breakdown["crop_health_reward"] = round(health_score * 0.3, 4)
        breakdown["efficiency_bonus"] = round(efficiency_score * 0.2, 4)

        forecasted = self.weather_sequence[min(self.weather_index, len(self.weather_sequence) - 1)]
        info["water_used"] = round(water_used, 4)
        info["fertilizer_used"] = round(fertilizer_used, 4)
        info["labor_used"] = round(labor_used, 4)
        info["forecasted_weather"] = forecasted
        info["actual_weather"] = actual_weather
        info["weather"] = actual_weather
        info["targeted_crop"] = crop.name.value
        info["market_prices"] = dict(self.market_prices)

        # Decision context logging for interpretability
        total_res = self.resources["water_units"] + self.resources["fertilizer_kg"] + self.resources["labor_hours"]
        best_market = max(self.market_prices.items(), key=lambda x: x[1])[0] if self.market_prices else "none"
        info["decision_context"] = {
            "priority_crop": crop.name.value,
            "risk_level": "high" if max_stress > 0.5 or actual_weather == "drought" else "medium" if max_stress > 0.2 else "low",
            "market_opportunity": best_market,
            "resource_pressure": "critical" if total_res < 10 else "tight" if total_res < 30 else "adequate",
            "weather_surprise": actual_weather != forecasted,
            "stress_alert": max_stress > 0.3,
        }

        # Advance day and weather
        self.day += 1
        self.weather_index += 1
        done = self.day >= 7

        # Final bonus on completion
        if done:
            total_yield = sum(expected_yield(c) for c in self.crops)
            total_profit_final = sum(expected_profit(c, self.market_prices) for c in self.crops)
            final_bonus = min(1.0, total_profit_final / 80.0)
            step_reward += final_bonus
            breakdown["final_yield_bonus"] = round(final_bonus, 4)
            info["total_yield"] = round(total_yield, 4)
            info["total_profit"] = round(total_profit_final, 4)

        self.cumulative_score += step_reward

        # Advance world state for next observation
        self._apply_stress_events()
        self.market_prices = self._generate_market_prices()
        self._actual_weather = self._sample_actual_weather()

        reward = FarmReward(
            value=round(step_reward, 4),
            breakdown=breakdown,
            done=done,
            info=info,
        )

        return self._get_observation(), reward, done, info

    def state(self) -> FarmObservation:
        if not self._initialized:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._get_observation()

    def _get_observation(self) -> FarmObservation:
        w_idx = min(self.weather_index, len(self.weather_sequence) - 1)
        forecast = self._generate_forecast(min(self.day, 7))
        return FarmObservation(
            day=min(self.day, 7),
            weather=WeatherCondition(self.weather_sequence[w_idx]),
            crops=[c.model_copy() for c in self.crops],
            resources=dict(self.resources),
            cumulative_score=round(self.cumulative_score, 4),
            task_description=self.task_description,
            forecast=forecast,
            market_prices=dict(self.market_prices),
        )
