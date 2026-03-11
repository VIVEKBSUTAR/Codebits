from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting Stage-3 tests...")

zone = "Bibwewadi"

# Baseline Risk
print("\n--- Baseline Risk ---")
response = client.get(f"/zone-risk?zone={zone}")
baseline = response.json()
print("Baseline:", baseline)

# Test 1 — Inject rainfall event
print("\n--- Test 1: Inject Rainfall Event ---")
response = client.post("/events", json={
    "event_type": "rainfall",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:00:00",
    "source": "weather_api"
})
assert response.status_code == 201

response = client.get(f"/zone-risk?zone={zone}")
post_rain_risk = response.json()
print("Post-Rainfall Risk:", post_rain_risk)
assert post_rain_risk["flood_risk"] > baseline["flood_risk"], "Flood risk should increase"

# Test 2 — Add construction event
print("\n--- Test 2: Add Construction Event ---")
response = client.post("/events", json={
    "event_type": "construction",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:10:00",
    "source": "city_api"
})
assert response.status_code == 201

response = client.get(f"/zone-risk?zone={zone}")
post_construction_risk = response.json()
print("Post-Construction Risk:", post_construction_risk)
assert post_construction_risk["traffic_risk"] > post_rain_risk["traffic_risk"], "Traffic risk should increase"

# Test 4 — Check causal explanation
print("\n--- Test 4: Check Causal Explanation ---")
response = client.get(f"/cause-analysis?zone={zone}&target=TrafficCongestion")
print("Response:", response.status_code, response.json())
assert response.status_code == 200

print("\nAll Stage-3 tests passed successfully!")
