from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting Stage-4 tests...")

zone = "Bibwewadi"

# Reset any internal state manually for fresh testing if necessary
print("\n--- Test 1: Rainfall Cascade ---")
response = client.post("/events", json={
    "event_type": "rainfall",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:00:00",
    "source": "weather_api"
})
assert response.status_code == 201

response = client.get(f"/zone-timeline?zone={zone}")
timeline_rain = response.json()
print("Rainfall Timeline:", timeline_rain)
assert len(timeline_rain["predicted_events"]) >= 3, "Should predict drainage, flooding, traffic"

# Find a predicted event
events = [e["event_name"] for e in timeline_rain["predicted_events"]]
assert "DrainageCapacity stress" in events
assert "Flooding" in events
assert "Traffic congestion" in events

print("\n--- Test 2: Construction Cascade ---")
# Post a construction event
response = client.post("/events", json={
    "event_type": "construction",
    "zone": zone,
    "severity": "high",
    "timestamp": "2026-04-12T09:10:00",
    "source": "city_api"
})
assert response.status_code == 201

# Run timeline again
response = client.get(f"/zone-timeline?zone={zone}")
timeline_combined = response.json()
print("Combined Timeline:", timeline_combined)
events_combined = [e["event_name"] for e in timeline_combined["predicted_events"]]
assert "Traffic congestion" in events_combined
assert "Emergency delays" in events_combined, "Emergency delays should now be elevated"

print("\nAll Stage-4 tests passed successfully!")
