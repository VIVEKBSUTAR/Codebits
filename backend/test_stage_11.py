import math
from fastapi.testclient import TestClient
from main import app
from analysis.prediction_confidence import get_edge_confidence
from causal_engine.causal_graph import get_causal_graph

client = TestClient(app)

print("Starting Stage-11 tests...")

zone = "Bibwewadi"

# Inject events (setup for Graph state)
client.post("/events", json={
    "event_type": "rainfall", "zone": zone, "severity": "high", "timestamp": "2026-04-12T09:00:00", "source": "api"
})

# TEST 1 - Edge Confidence
print("\n--- Test 1: Edge Confidence ---")
val_rain_flood = get_edge_confidence("Rainfall", "Flooding")
assert math.isclose(val_rain_flood, 342 / 432, rel_tol=1e-2), f"Expected 0.79, got {val_rain_flood}"

# TEST 2 - Missing Evidence
print("\n--- Test 2: Missing Evidence ---")
val_missing = get_edge_confidence("Unknown", "Flooding")
assert val_missing == 0.5, f"Expected 0.5, got {val_missing}"


# TEST 4 - API Output and TEST 3/5
print("\n--- Test 4: API Output & Test 3 Cascade & Test 5 Normalization ---")
response = client.get(f"/zone-risk?zone={zone}")
print("Response Status:", response.status_code)
assert response.status_code == 200

data = response.json()
print("Risk Response:", data)

assert "zone" in data
assert "predictions" in data

predictions = data["predictions"]
assert len(predictions) >= 3

found_flood = False
found_traffic = False

for item in predictions:
    assert "event" in item
    assert "probability" in item
    assert "confidence" in item
    
    # TEST 5 - Normalization check
    conf = item["confidence"]
    assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of bounds"
    
    if item["event"] == "Flooding":
        found_flood = True
        flood_conf = conf
    elif item["event"] == "TrafficCongestion":
        found_traffic = True
        traffic_conf = conf

assert found_flood and found_traffic

# TEST 3 - Cascade Confidence
# Rainfall -> Flooding -> TrafficCongestion
# Therefore confidence of TrafficCongestion < Flooding 
# (Because multiplication of fractions reduces the product)
assert traffic_conf <= flood_conf, f"Cascade failed: Traffic ({traffic_conf}) should have lower confidence than Flooding ({flood_conf})"

print("\nAll Stage-11 tests passed successfully!")
