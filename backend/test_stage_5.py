from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting Stage-5 tests...")

zone = "Bibwewadi"

# Baseline Setup: Post a rainfall event
print("\n--- Setup: Post Rainfall ---")
response = client.post("/events", json={
    "event_type": "rainfall",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:00:00",
    "source": "weather_api"
})
assert response.status_code == 201

# Test 1 -- Pump Intervention
print("\n--- Test 1: Pump Intervention ---")
response = client.post("/simulate-intervention", json={
    "zone": zone,
    "intervention": "deploy_pump"
})
print("Pump Response:", response.status_code, response.json())
assert response.status_code == 200
data = response.json()
assert data["benefit"]["flooding_reduction"] > 0, "Pump should reduce flooding risk"

# Baseline Setup: Post construction event
print("\n--- Setup: Post Construction ---")
response = client.post("/events", json={
    "event_type": "construction",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:10:00",
    "source": "city_api"
})
assert response.status_code == 201

# Test 2 -- Road Closure
print("\n--- Test 2: Road Closure ---")
response = client.post("/simulate-intervention", json={
    "zone": zone,
    "intervention": "close_road"
})
print("Road Closure Response:", response.status_code, response.json())
assert response.status_code == 200
data = response.json()
assert data["benefit"]["traffic_reduction"] > 0, "Road closure should reduce traffic risk"

# Test 3 -- Ambulance Dispatch
print("\n--- Test 3: Ambulance Dispatch ---")
response = client.post("/simulate-intervention", json={
    "zone": zone,
    "intervention": "dispatch_ambulance"
})
print("Ambulance Response:", response.status_code, response.json())
assert response.status_code == 200
data = response.json()
assert data["benefit"]["emergency_delay_reduction"] > 0, "Ambulance should reduce delays"

print("\nAll Stage-5 tests passed successfully!")
