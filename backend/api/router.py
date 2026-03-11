from fastapi import APIRouter

from api.health_api import router as health_router
from api.events_api import router as events_router
from api.causal_api import router as causal_router

main_router = APIRouter()

# Attach individual domain routers
main_router.include_router(health_router)
main_router.include_router(events_router)
main_router.include_router(causal_router)



@main_router.get("/predictions")
async def get_predictions():
    return {"status": "stub", "message": "Prediction endpoint not implemented yet."}

@main_router.get("/interventions")
async def get_interventions():
    return {"status": "stub", "message": "Interventions endpoint not implemented yet."}
