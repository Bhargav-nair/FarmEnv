from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env import FarmEnv, FarmAction, FarmObservation, FarmReward
from tasks import TASK_SCENARIOS

app = FastAPI(
    title="FarmEnv",
    version="1.0.0",
    description="An advanced farm resource allocation environment where an AI agent optimizes profit under weather uncertainty, crop stress events, and dynamic market conditions.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = FarmEnv()


class ResetRequest(BaseModel):
    task: Optional[str] = None


class StepResponse(BaseModel):
    observation: FarmObservation
    reward: FarmReward
    done: bool
    info: dict


@app.on_event("startup")
async def startup():
    env.reset()


@app.post("/reset", response_model=FarmObservation)
async def reset(request: ResetRequest = ResetRequest()):
    global env
    scenario = None
    if request.task:
        if request.task not in TASK_SCENARIOS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown task: {request.task}. Available: {list(TASK_SCENARIOS.keys())}",
            )
        scenario = TASK_SCENARIOS[request.task]
    env = FarmEnv(scenario=scenario)
    return env.reset()


@app.post("/step", response_model=StepResponse)
async def step(action: FarmAction):
    try:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=FarmObservation)
async def get_state():
    try:
        return env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/tasks")
async def get_tasks():
    return [
        {
            "id": "task1",
            "name": "Single Crop Water Management",
            "difficulty": "easy",
            "max_steps": 3,
            "description": "Manage water levels for a single wheat crop over 3 days. Keep water_level between 0.4-0.7.",
        },
        {
            "id": "task2",
            "name": "Multi-Crop Triage Under Scarcity",
            "difficulty": "medium",
            "max_steps": 5,
            "description": "Save a stressed corn crop under resource scarcity, weather uncertainty, and stress events while maintaining other crops.",
        },
        {
            "id": "task3",
            "name": "Week-Long Yield Optimization",
            "difficulty": "hard",
            "max_steps": 7,
            "description": "Maximize total profit across crops under weather uncertainty, stress events, and dynamic market conditions.",
        },
    ]


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "FarmEnv"}

@app.get("/")
async def root():
    return {
        "message": "FarmEnv API is running",
        "endpoints": {
            "health": "/health",
            "reset": "/reset",
            "step": "/step",
            "state": "/state",
            "tasks": "/tasks"
        }
    }

@app.get("/reset")
def reset_get():
    return {"message": "Use POST /reset"}
