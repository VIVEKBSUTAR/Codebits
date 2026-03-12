"""
Live data fetcher service.
Pulls real-time weather (OpenWeatherMap) and traffic (TomTom) data,
stores it to the database, and converts it into time-series rows
for the causal discovery algorithms.
"""
import os
import json
import asyncio
import httpx
from datetime import datetime
from typing import Dict, Optional
from database import db
from utils.logger import SystemLogger

logger = SystemLogger(module_name="live_data")

# ── API Keys (set via environment variables) ─────────────────────────────────
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "")

# Zone coordinates for Pune
ZONE_COORDS = {
    "Bibwewadi":   {"lat": 18.4582, "lon": 73.8620},
    "Katraj":      {"lat": 18.4437, "lon": 73.8638},
    "Hadapsar":    {"lat": 18.4980, "lon": 73.9258},
    "Kothrud":     {"lat": 18.5074, "lon": 73.8077},
    "Warje":       {"lat": 18.4832, "lon": 73.8118},
    "Sinhagad":    {"lat": 18.4326, "lon": 73.8095},
    "Pimpri":      {"lat": 18.6279, "lon": 73.8009},
    "Chinchwad":   {"lat": 18.6261, "lon": 73.7828},
    "Hinjewadi":   {"lat": 18.5913, "lon": 73.7389},
    "Wakad":       {"lat": 18.5987, "lon": 73.7688},
    "Baner":       {"lat": 18.5590, "lon": 73.7868},
    "Shivajinagar":{"lat": 18.5314, "lon": 73.8446},
}


async def fetch_weather(zone: str) -> Optional[Dict]:
    """Fetch current weather from OpenWeatherMap for a zone."""
    if not OPENWEATHER_API_KEY:
        return None
    coords = ZONE_COORDS.get(zone)
    if not coords:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            rain_1h = data.get("rain", {}).get("1h", 0)
            parsed = {
                "timestamp": datetime.now().isoformat(),
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "rainfall_mm": rain_1h,
                "wind_speed": data["wind"]["speed"],
                "condition": data["weather"][0]["main"] if data.get("weather") else "Unknown",
                "source": "openweathermap",
            }
            db.store_weather(zone, parsed)
            logger.log(f"Weather stored for {zone}: {parsed['condition']}, rain={rain_1h}mm")
            return parsed
    except Exception as e:
        logger.log(f"Weather fetch failed for {zone}: {e}")
        return None


async def fetch_traffic(zone: str) -> Optional[Dict]:
    """Fetch traffic flow from TomTom for a zone."""
    if not TOMTOM_API_KEY:
        return None
    coords = ZONE_COORDS.get(zone)
    if not coords:
        return None
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{coords['lat']},{coords['lon']}",
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            flow = data.get("flowSegmentData", {})
            free_flow = flow.get("freeFlowSpeed", 60)
            current = flow.get("currentSpeed", 60)
            congestion = max(0, min(1, 1 - (current / max(free_flow, 1))))
            parsed = {
                "timestamp": datetime.now().isoformat(),
                "congestion_level": round(congestion, 3),
                "average_speed": current,
                "incident_count": 0,
                "road_segment": flow.get("frc", "unknown"),
                "source": "tomtom",
            }
            db.store_traffic(zone, parsed)
            logger.log(f"Traffic stored for {zone}: congestion={congestion:.1%}")
            return parsed
    except Exception as e:
        logger.log(f"Traffic fetch failed for {zone}: {e}")
        return None


async def fetch_all_zones():
    """Fetch weather and traffic for all zones in parallel."""
    tasks = []
    for zone in ZONE_COORDS:
        tasks.append(fetch_weather(zone))
        tasks.append(fetch_traffic(zone))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if r is not None and not isinstance(r, Exception))
    logger.log(f"Fetched {success}/{len(tasks)} data points across {len(ZONE_COORDS)} zones")
    return success


def aggregate_to_timeseries(zone: str):
    """
    Aggregate latest weather + traffic data into a timeseries row
    for causal discovery algorithms.
    """
    weather = db.get_weather(zone, hours=1)
    traffic = db.get_traffic(zone, hours=1)

    rainfall = 0
    if weather:
        latest_w = weather[-1]
        rainfall = latest_w.get("rainfall_mm", 0)

    congestion = 0
    if traffic:
        latest_t = traffic[-1]
        congestion = latest_t.get("congestion_level", 0)

    # Derive drainage load from rainfall intensity
    drainage_load = min(1.0, rainfall / 50.0)  # 50mm = full load
    # Derive flooding level from rainfall + drainage
    flooding_level = max(0, (rainfall / 100.0) + drainage_load * 0.3 - 0.2)
    flooding_level = min(1.0, flooding_level)
    # Emergency delay correlates with congestion + flooding
    emergency_delay = min(1.0, congestion * 0.4 + flooding_level * 0.5)

    ts_data = {
        "timestamp": datetime.now().isoformat(),
        "rainfall": rainfall,
        "drainage_load": round(drainage_load, 3),
        "flooding_level": round(flooding_level, 3),
        "traffic_congestion": round(congestion, 3),
        "emergency_delay": round(emergency_delay, 3),
        "construction_activity": 0,
        "accident_count": 0,
    }
    db.store_timeseries(zone, ts_data)
    return ts_data


async def periodic_fetch(interval_minutes: int = 15):
    """Background task: fetch live data every N minutes."""
    while True:
        try:
            await fetch_all_zones()
            for zone in ZONE_COORDS:
                aggregate_to_timeseries(zone)
        except Exception as e:
            logger.log(f"Periodic fetch error: {e}")
        await asyncio.sleep(interval_minutes * 60)
