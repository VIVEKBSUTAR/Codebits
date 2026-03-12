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

    # Initialize database
    from database.db import init_db
    init_db()
    logger.log("Database initialized")

    # Start background live data fetch (if API keys configured)
    import asyncio
    from ingestion.live_data import periodic_fetch, OPENWEATHER_API_KEY, TOMTOM_API_KEY
    if OPENWEATHER_API_KEY or TOMTOM_API_KEY:
        asyncio.create_task(periodic_fetch(interval_minutes=15))
        logger.log("Live data background fetch started (15min interval)")
    else:
        logger.log("No live API keys configured — set OPENWEATHER_API_KEY / TOMTOM_API_KEY")

    yield
    # Shutdown (nothing to clean up)

app = FastAPI(title="City-Scale Event Causality Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://ai-city-management.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
