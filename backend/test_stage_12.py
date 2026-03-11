from fastapi.testclient import TestClient
from main import app
import time

client = TestClient(app)

print("Starting Stage-12 tests...")

zone = "Bibwewadi_Auditable"

# TEST 1 - Incident Creation
print("\n--- Test 1: Incident Creation ---")
response = client.post("/events", json={
    "event_type": "rainfall", "zone": zone, "severity": "high", "timestamp": "2026-04-12T09:00:00", "source": "weather_api"
})
assert response.status_code == 201

# TEST 2 - Prediction Logging 
print("\n--- Test 2: Prediction Logging ---")
response = client.get(f"/zone-risk?zone={zone}")
assert response.status_code == 200

# TEST 3 - Action Logging
print("\n--- Test 3: Action Logging ---")
# Let's prompt the optimizer which should log a recommendation
response = client.get(f"/optimal-deployment?resources=pumps:1")
assert response.status_code == 200

# Let's simulate intervention which logs the decision action
response = client.post("/simulate-intervention", json={
    "zone": zone,
    "intervention": "deploy_pump"
})
assert response.status_code == 200

# TEST 4 - Incident Retrieval
print("\n--- Test 4 & 5: Incident Retrieval & Data Integrity ---")
response = client.get(f"/incident-history?zone={zone}")
print("History Response:", response.status_code, response.json())
assert response.status_code == 200

data = response.json()
assert data["zone"] == zone
incidents = data["incidents"]
assert len(incidents) >= 1

latest_incident = incidents[0]

# TEST 5 - Data Integrity
assert latest_incident["incident_id"].startswith("INC_")
assert latest_incident["zone"] == zone
assert latest_incident["event_type"] == "rainfall"
# Confirm prediction was attached
assert latest_incident["prediction"] is not None
# E.g. {"Flooding": {"probability": 0.72, "confidence": 0.79}}
assert "Flooding" in latest_incident["prediction"]
assert latest_incident["prediction"]["Flooding"]["probability"] > 0
assert latest_incident["prediction"]["Flooding"]["confidence"] > 0

# Confirm recommendation was attached
assert latest_incident["recommended_action"] == "deploy_pump"

# Confirm human decision and action was attached
assert latest_incident["human_decision"] == "approved"
assert latest_incident["final_action"] == "deploy_pump"

print("\nAll Stage-12 tests passed successfully!")
