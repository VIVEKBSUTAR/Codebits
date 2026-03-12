"""
Common Attribute Space for Causal Discovery.

Defines the unified set of city event attributes that ALL causal discovery
algorithms (Granger, PCMCI, NOTEARS) operate on. Any attribute added here
becomes automatically available to every algorithm.

Categories:
- Weather: rainfall, temperature, humidity, wind_speed
- Infrastructure: drainage_load, power_grid_load, water_supply_pressure
- Incidents: flooding_level, accident_count, power_outage, fire_incident
- Traffic: traffic_congestion, emergency_delay
- Environment: heatwave_index, air_quality_index
- Human: construction_activity, public_event_crowd, industrial_discharge
"""
import numpy as np
from typing import Dict, List, Tuple, Optional

# ── Master Attribute Registry ────────────────────────────────────────────────
# Every attribute the system tracks. All 3 algorithms pull from this list.

ATTRIBUTES: List[Dict] = [
    # Weather
    {"name": "rainfall",              "category": "weather",        "unit": "mm/h",    "desc": "Precipitation intensity"},
    {"name": "temperature",           "category": "weather",        "unit": "°C",      "desc": "Ambient temperature"},
    {"name": "humidity",              "category": "weather",        "unit": "%",       "desc": "Relative humidity"},
    {"name": "wind_speed",            "category": "weather",        "unit": "m/s",     "desc": "Wind speed"},
    # Infrastructure
    {"name": "drainage_load",         "category": "infrastructure", "unit": "ratio",   "desc": "Storm drain utilization (0-1)"},
    {"name": "power_grid_load",       "category": "infrastructure", "unit": "ratio",   "desc": "Electrical grid load factor (0-1)"},
    {"name": "water_supply_pressure", "category": "infrastructure", "unit": "bar",     "desc": "Municipal water pressure"},
    # Incidents
    {"name": "flooding_level",        "category": "incident",       "unit": "ratio",   "desc": "Area flood severity (0-1)"},
    {"name": "accident_count",        "category": "incident",       "unit": "count",   "desc": "Road accidents in period"},
    {"name": "power_outage",          "category": "incident",       "unit": "binary",  "desc": "Active power outage (0/1)"},
    {"name": "fire_incident",         "category": "incident",       "unit": "binary",  "desc": "Active fire incident (0/1)"},
    # Traffic
    {"name": "traffic_congestion",    "category": "traffic",        "unit": "ratio",   "desc": "Road congestion level (0-1)"},
    {"name": "emergency_delay",       "category": "traffic",        "unit": "ratio",   "desc": "Emergency response delay factor (0-1)"},
    # Environment
    {"name": "heatwave_index",        "category": "environment",    "unit": "index",   "desc": "Heat stress index (temp × humidity factor)"},
    {"name": "air_quality_index",     "category": "environment",    "unit": "AQI",     "desc": "Air Quality Index (0-500)"},
    # Human activity
    {"name": "construction_activity", "category": "human",          "unit": "ratio",   "desc": "Construction disruption level (0-1)"},
    {"name": "public_event_crowd",    "category": "human",          "unit": "ratio",   "desc": "Large public gathering intensity (0-1)"},
    {"name": "industrial_discharge",  "category": "human",          "unit": "ratio",   "desc": "Industrial water/waste discharge level (0-1)"},
]

# Flat list of attribute names — this is what the algorithms use
ATTRIBUTE_NAMES: List[str] = [a["name"] for a in ATTRIBUTES]

# Lookup by name
ATTRIBUTE_MAP: Dict[str, Dict] = {a["name"]: a for a in ATTRIBUTES}

# Known causal priors (expected relationships for the unknown-cause engine)
KNOWN_CAUSES: Dict[str, List[str]] = {
    "flooding_level":       ["rainfall", "drainage_load", "industrial_discharge"],
    "traffic_congestion":   ["flooding_level", "construction_activity", "accident_count", "public_event_crowd"],
    "emergency_delay":      ["traffic_congestion", "flooding_level", "power_outage"],
    "drainage_load":        ["rainfall", "industrial_discharge"],
    "power_outage":         ["power_grid_load", "heatwave_index", "flooding_level"],
    "fire_incident":        ["heatwave_index", "power_outage", "industrial_discharge"],
    "heatwave_index":       ["temperature", "humidity"],
    "air_quality_index":    ["traffic_congestion", "industrial_discharge", "construction_activity", "wind_speed"],
    "accident_count":       ["traffic_congestion", "rainfall", "heatwave_index"],
}


def get_attribute_names(categories: Optional[List[str]] = None) -> List[str]:
    """Return attribute names, optionally filtered by category."""
    if categories is None:
        return ATTRIBUTE_NAMES
    return [a["name"] for a in ATTRIBUTES if a["category"] in categories]


def get_attributes_info() -> List[Dict]:
    """Return full attribute metadata for UI display."""
    return ATTRIBUTES


def prepare_matrix(timeseries: List[Dict],
                   variables: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Convert list-of-dicts timeseries into numpy matrix (T x V).
    Uses the common attribute space. Skips constant columns.
    This is the SINGLE data preparation function all algorithms should use.
    """
    vars_to_use = variables or ATTRIBUTE_NAMES
    cols = []
    valid_vars = []
    for v in vars_to_use:
        col = [float(row.get(v, 0)) for row in timeseries]
        if np.std(col) > 1e-8:  # skip constant columns
            cols.append(col)
            valid_vars.append(v)
    if not cols:
        return np.array([]), []
    return np.array(cols).T, valid_vars


def prepare_matrix_standardized(timeseries: List[Dict],
                                variables: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Same as prepare_matrix but with z-score standardization.
    Used by NOTEARS which benefits from normalized data.
    """
    vars_to_use = variables or ATTRIBUTE_NAMES
    cols = []
    valid_vars = []
    for v in vars_to_use:
        col = [float(row.get(v, 0)) for row in timeseries]
        if np.std(col) > 1e-8:
            cols.append(col)
            valid_vars.append(v)
    if not cols:
        return np.array([]), []
    mat = np.array(cols).T
    means = mat.mean(axis=0)
    stds = mat.std(axis=0)
    stds[stds < 1e-10] = 1.0
    mat = (mat - means) / stds
    return mat, valid_vars
