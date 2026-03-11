from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("Starting Stage-6 tests...")

# 1. Setup Active Zones with varying events
print("\n--- Setup: Initializing Events Across Zones ---")

# Zone 1: Katraj (Construction only)
client.post("/events", json={
    "event_type": "construction", "zone": "Katraj", "severity": "high", "timestamp": "2026-04-12T09:00:00", "source": "api"
})

# Zone 2: Shivajinagar (Accident)
client.post("/events", json={
    "event_type": "accident", "zone": "Shivajinagar", "severity": "high", "timestamp": "2026-04-12T09:05:00", "source": "api"
})

# Zone 3: Bibwewadi (Rainfall - Highest cascading risk)
client.post("/events", json={
    "event_type": "rainfall", "zone": "Bibwewadi", "severity": "high", "timestamp": "2026-04-12T09:10:00", "source": "api"
})


# Test 1 -- Pump Allocation
print("\n--- Test 1: Allocating 1 Pump ---")
# 3 zones flooded/stressed, but Bibwewadi has the most risk 
response = client.get("/optimal-deployment?resources=pumps:1")
print("Pump Response:", response.json())
assert response.status_code == 200
data = response.json()
assert len(data["plan"]) == 1
assert data["plan"][0]["resource"] == "pump"
# Bibwewadi rainfall creates highest flood risk
assert data["plan"][0]["zone"] == "Bibwewadi"


# Test 2 -- Multi-Resource Distribution
print("\n--- Test 2: Allocating Mixed Resources ---")
response = client.get("/optimal-deployment?resources=pumps:1,ambulances:2,traffic:1")
print("Mixed Response:", response.json())
assert response.status_code == 200
data = response.json()
# We expect multiple assignments across the city based on greedy sorting
assert len(data["plan"]) <= 4
assert data["expected_citywide_risk_reduction"] > 0

# Check that the constraint wasn't exceeded
pump_count = sum(1 for item in data["plan"] if item["resource"] == "pump")
assert pump_count <= 1

ambulance_count = sum(1 for item in data["plan"] if item["resource"] == "ambulance")
assert ambulance_count <= 2
    
print("\nAll Stage-6 tests passed successfully!")
