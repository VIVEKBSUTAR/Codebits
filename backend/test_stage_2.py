from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting tests...")

# Test 1 - Submit event
print("--- Test 1: Submit Event ---")
response = client.post("/events", json={
    "event_type": "rainfall",
    "zone": "Bibwewadi",
    "severity": "high",
    "timestamp": "2026-04-12T09:00:00",
    "source": "weather_api",
    "metadata": {"rain_intensity": "20mm"}
})
print("Response:", response.status_code, response.json())
assert response.status_code == 201
assert response.json()["status"] == "event_received"
assert "event_id" in response.json()

# Test 2 - Retrieve events
print("\n--- Test 2: Retrieve Events ---")
response = client.get("/events")
print("Response:", response.status_code, response.json())
assert response.status_code == 200
assert len(response.json()) >= 1

# Test 3 - Zone filter
print("\n--- Test 3: Zone Filter ---")
response = client.get("/events?zone=Bibwewadi")
print("Response:", response.status_code, response.json())
assert response.status_code == 200
assert response.json()[0]["zone"] == "Bibwewadi"

# Test 4 - Validation error
print("\n--- Test 4: Validation Error ---")
response = client.post("/events", json={
    "event_type": "unknown_event"
})
print("Response:", response.status_code, response.json())
assert response.status_code == 422

print("\nAll tests passed successfully!")
