import math
from fastapi.testclient import TestClient
from main import app
from analysis.confidence_engine import get_confidence

client = TestClient(app)

print("Starting Stage-10 tests...")

zone = "Bibwewadi"

# Inject events so that parents have active probabilities
client.post("/events", json={
    "event_type": "rainfall", "zone": zone, "severity": "high", "timestamp": "2026-04-12T09:00:00", "source": "api"
})
client.post("/events", json={
    "event_type": "construction", "zone": zone, "severity": "high", "timestamp": "2026-04-12T09:10:00", "source": "api"
})
# Missing Accident event on purpose to ensure its probability is computed as lower.

# TEST 3 - Confidence Matrix Usage
print("\n--- Test 3: Confidence Matrix Usage ---")
c_val = get_confidence("ConstructionActivity", "TrafficCongestion")
assert c_val == 0.91, f"Expected 0.91, got {c_val}"

c_val_unknown = get_confidence("Unknown", "TrafficCongestion")
assert c_val_unknown == 0.5, f"Expected 0.5, got {c_val_unknown}"

# TEST 5 - API Endpoint and Tests 1, 2, 4 checks
print("\n--- Test 5: API Endpoint ---")
response = client.get(f"/cause-analysis?zone={zone}&event=TrafficCongestion")
print("Analysis Response:", response.status_code, response.json())
assert response.status_code == 200

data = response.json()
assert data["zone"] == zone
assert data["event"] == "TrafficCongestion"

causes = data["causes"]
# TEST 1 - Cause Identification
parent_names = [c["event"] for c in causes]
assert "Flooding" in parent_names
assert "ConstructionActivity" in parent_names
assert "Accident" in parent_names

# TEST 2 - Contribution Ranking
# Because we injected ConstructionActivity, its P=1.0. 
# Flooding is partial depending on Rainfall/Drainage. Accident is baseline low.
# Construction should be ranked above Accident.
construction_c = next(c for c in causes if c["event"] == "ConstructionActivity")
accident_c = next(c for c in causes if c["event"] == "Accident")
assert construction_c["contribution"] > accident_c["contribution"]

# TEST 4 - Normalization check
total_contrib = sum(c["contribution"] for c in causes)
assert math.isclose(total_contrib, 1.0, rel_tol=1e-2), f"Contributions do not sum to 1.0, total={total_contrib}"

print("\nAll Stage-10 tests passed successfully!")
