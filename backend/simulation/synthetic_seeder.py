"""
Synthetic Data Seeder for Demo/Judging.

Generates realistic time-series data with KNOWN causal relationships
embedded so that Granger, PCMCI, and NOTEARS will discover them.

The data follows the causal priors defined in attribute_space.py:
  rainfall → drainage_load → flooding_level → traffic_congestion → emergency_delay
  temperature + humidity → heatwave_index → power_outage → fire_incident
  traffic_congestion + industrial_discharge → air_quality_index
  etc.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from causal_engine.attribute_space import ATTRIBUTE_NAMES


def generate_synthetic_timeseries(
    zone: str = "Bibwewadi",
    hours: int = 168,
    seed: Optional[int] = 42,
) -> List[Dict]:
    """
    Generate synthetic time-series that encodes known causal relationships.

    The data is designed so that:
    - Granger will find lagged effects (rainfall → flooding after 2h)
    - PCMCI will find conditional independence structure
    - NOTEARS will recover the DAG structure
    - Some targets (power_outage, fire_incident) will have WEAK signal so
      the AI prediction layer gets activated as fallback.

    Returns list of dicts, one per hour, with all 18 attributes.
    """
    if seed is not None:
        np.random.seed(seed)

    T = hours
    data = []
    now = datetime.now()

    # ── Base exogenous signals ──────────────────────────────────────────────
    # These are independent root causes
    rainfall = np.clip(
        np.cumsum(np.random.randn(T) * 0.3) * 0.15 + 2.0 + 3.0 * np.sin(np.linspace(0, 4 * np.pi, T)),
        0, 25
    )
    temperature = 28 + 6 * np.sin(np.linspace(0, 6 * np.pi, T)) + np.random.randn(T) * 1.5
    humidity = np.clip(55 + 15 * np.sin(np.linspace(np.pi, 5 * np.pi, T)) + np.random.randn(T) * 5, 20, 95)
    wind_speed = np.clip(np.abs(3 + np.random.randn(T) * 2), 0, 15)
    construction_activity = np.clip(0.3 + 0.2 * np.sin(np.linspace(0, 2 * np.pi, T)) + np.random.randn(T) * 0.08, 0, 1)
    public_event_crowd = np.zeros(T)
    # Inject 2-3 public events
    for start in [T // 5, T // 2, 4 * T // 5]:
        dur = min(8, T - start)
        public_event_crowd[start:start + dur] = np.clip(0.6 + np.random.rand(dur) * 0.3, 0, 1)
    industrial_discharge = np.clip(0.15 + np.random.randn(T) * 0.06 + 0.1 * np.sin(np.linspace(0, 3 * np.pi, T)), 0, 1)

    # ── Derived signals with causal lags ────────────────────────────────────

    # drainage_load = f(rainfall[t-1], industrial_discharge[t-1]) + noise
    drainage_load = np.zeros(T)
    for t in range(1, T):
        drainage_load[t] = np.clip(
            0.15 + 0.03 * rainfall[t - 1] + 0.3 * industrial_discharge[t - 1] + np.random.randn() * 0.04,
            0, 1
        )

    # power_grid_load = base + temperature effect
    power_grid_load = np.clip(
        0.4 + 0.012 * (temperature - 25) + 0.1 * np.sin(np.linspace(0, 4 * np.pi, T)) + np.random.randn(T) * 0.05,
        0.1, 1
    )

    # water_supply_pressure = base (inversely related to drainage_load during flooding)
    water_supply_pressure = np.clip(
        2.5 - 0.8 * drainage_load + np.random.randn(T) * 0.15,
        0.5, 4.0
    )

    # heatwave_index = f(temperature, humidity) — strong known relationship
    heatwave_index = np.clip(
        0.015 * temperature * (humidity / 100.0) + np.random.randn(T) * 0.05,
        0, 2.0
    )

    # flooding_level = f(rainfall[t-2], drainage_load[t-1]) — lagged
    flooding_level = np.zeros(T)
    for t in range(2, T):
        flooding_level[t] = np.clip(
            0.04 * rainfall[t - 2] + 0.6 * drainage_load[t - 1] + np.random.randn() * 0.03,
            0, 1
        )

    # accident_count = f(traffic_congestion[t-1], rainfall[t], heatwave_index[t])
    # (traffic_congestion not yet computed — we'll do 2 passes)
    accident_count = np.zeros(T)

    # traffic_congestion = f(flooding_level[t-1], construction_activity, accident_count[t-1], public_event_crowd)
    traffic_congestion = np.zeros(T)
    for t in range(1, T):
        traffic_congestion[t] = np.clip(
            0.2 + 0.35 * flooding_level[t - 1] + 0.25 * construction_activity[t] +
            0.2 * public_event_crowd[t] + np.random.randn() * 0.05,
            0, 1
        )

    # Now compute accident_count with traffic_congestion available
    for t in range(1, T):
        accident_count[t] = max(0, round(
            0.5 + 2.5 * traffic_congestion[t - 1] + 0.15 * rainfall[t] +
            0.8 * heatwave_index[t] + np.random.randn() * 0.4
        ))

    # Re-compute traffic_congestion with accident feedback
    for t in range(2, T):
        traffic_congestion[t] = np.clip(
            0.2 + 0.35 * flooding_level[t - 1] + 0.25 * construction_activity[t] +
            0.15 * (accident_count[t - 1] / max(np.max(accident_count), 1)) +
            0.2 * public_event_crowd[t] + np.random.randn() * 0.04,
            0, 1
        )

    # emergency_delay = f(traffic_congestion[t], flooding_level[t], power_outage[t])
    # power_outage needs to be computed first

    # power_outage = f(power_grid_load[t-1], heatwave_index[t], flooding_level[t])
    # Make this WEAK — so AI prediction layer gets triggered for some zones
    power_outage = np.zeros(T)
    for t in range(1, T):
        prob = 0.02 + 0.08 * power_grid_load[t - 1] + 0.05 * heatwave_index[t] + 0.04 * flooding_level[t]
        power_outage[t] = 1.0 if np.random.rand() < prob else 0.0

    # fire_incident = f(heatwave_index, power_outage, industrial_discharge) — VERY WEAK
    fire_incident = np.zeros(T)
    for t in range(T):
        prob = 0.01 + 0.03 * heatwave_index[t] + 0.02 * power_outage[t] + 0.02 * industrial_discharge[t]
        fire_incident[t] = 1.0 if np.random.rand() < prob else 0.0

    # emergency_delay = f(traffic_congestion, flooding_level, power_outage)
    emergency_delay = np.zeros(T)
    for t in range(T):
        emergency_delay[t] = np.clip(
            0.1 + 0.4 * traffic_congestion[t] + 0.3 * flooding_level[t] +
            0.15 * power_outage[t] + np.random.randn() * 0.04,
            0, 1
        )

    # air_quality_index = f(traffic_congestion, industrial_discharge, construction_activity, wind_speed)
    air_quality_index = np.clip(
        50 + 120 * traffic_congestion + 80 * industrial_discharge +
        40 * construction_activity - 8 * wind_speed + np.random.randn(T) * 15,
        0, 500
    )

    # ── Package into list of dicts ──────────────────────────────────────────
    for t in range(T):
        ts = (now - timedelta(hours=T - t)).isoformat()
        row = {
            "zone": zone,
            "timestamp": ts,
            "rainfall": round(float(rainfall[t]), 2),
            "temperature": round(float(temperature[t]), 2),
            "humidity": round(float(humidity[t]), 2),
            "wind_speed": round(float(wind_speed[t]), 2),
            "drainage_load": round(float(drainage_load[t]), 3),
            "power_grid_load": round(float(power_grid_load[t]), 3),
            "water_supply_pressure": round(float(water_supply_pressure[t]), 2),
            "flooding_level": round(float(flooding_level[t]), 3),
            "accident_count": round(float(accident_count[t]), 0),
            "power_outage": float(power_outage[t]),
            "fire_incident": float(fire_incident[t]),
            "traffic_congestion": round(float(traffic_congestion[t]), 3),
            "emergency_delay": round(float(emergency_delay[t]), 3),
            "heatwave_index": round(float(heatwave_index[t]), 3),
            "air_quality_index": round(float(air_quality_index[t]), 1),
            "construction_activity": round(float(construction_activity[t]), 3),
            "public_event_crowd": round(float(public_event_crowd[t]), 3),
            "industrial_discharge": round(float(industrial_discharge[t]), 3),
        }
        data.append(row)

    return data


def seed_zone(zone: str = "Bibwewadi", hours: int = 168, seed: Optional[int] = 42) -> int:
    """Generate synthetic data and store it in the database."""
    from database import db
    rows = generate_synthetic_timeseries(zone=zone, hours=hours, seed=seed)
    count = 0
    for row in rows:
        db.store_timeseries(zone, row)
        count += 1
    return count
