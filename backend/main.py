from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import main_router
from utils.logger import SystemLogger

logger = SystemLogger(module_name="server")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.log("backend initialized")
    import ingestion.event_processor
    logger.log("Event ingestion module loaded")
    import causal_engine.causal_graph
    logger.log("Causal engine module initialized")
    import simulation.cascade_predictor
    logger.log("Simulation engine initialized")
    import optimization.resource_optimizer
    logger.log("Optimization module ready")
    yield
    # Shutdown (nothing to clean up)

app = FastAPI(title="City-Scale Event Causality Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
