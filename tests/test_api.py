"""Smoke/integration tests for the FastAPI return prediction API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

payload = {
    "product_price": 2499.0,
    "product_rating": 3.2,
    "product_historical_return_rate": 0.35,
    "customer_age": 31,
    "customer_previous_orders": 8,
    "customer_previous_returns": 3,
    "customer_return_rate": 0.375,
    "quantity": 1,
    "discount_pct": 25.0,
    "delivery_distance_km": 18.0,
    "expected_delivery_days": 4,
    "order_month": 10,
    "order_day_of_week": 2,
    "category": "Fashion",
    "payment_type": "COD",
    "discount_bucket": "Medium",
    "price_bucket": "Mid",
}

health = client.get("/health")
assert health.status_code == 200, health.text
assert health.json()["status"] == "ok"

response = client.post("/predict", json=payload)
assert response.status_code == 200, response.text
data = response.json()

required = [
    "return_probability", "risk_level", "recommended_action",
    "intervention_threshold", "expected_avoided_return_cost",
    "intervention_cost", "net_expected_value", "explanation",
]
for field in required:
    assert field in data, f"Missing field: {field}"

assert 0 <= data["return_probability"] <= 1
assert data["recommended_action"] in {"INTERVENE", "MONITOR", "NO_ACTION"}
assert data["explanation"]["recommended_action"] == data["recommended_action"]

print("API smoke test: PASS")
print("Health:", health.json())
print("Prediction:")
print(data)
