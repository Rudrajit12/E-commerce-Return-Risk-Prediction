"""Pydantic request/response schemas for the Return Intelligence API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrderRequest(BaseModel):
    product_price: float = Field(ge=0)
    product_rating: float = Field(ge=0, le=5)
    product_historical_return_rate: float = Field(ge=0, le=1)
    customer_age: int = Field(ge=18, le=100)
    customer_previous_orders: int = Field(ge=0)
    customer_previous_returns: int = Field(ge=0)
    customer_return_rate: float = Field(ge=0, le=1)
    quantity: int = Field(ge=1)
    discount_pct: float = Field(ge=0, le=100)
    delivery_distance_km: float = Field(ge=0)
    expected_delivery_days: int = Field(ge=1)
    order_month: int = Field(ge=1, le=12)
    order_day_of_week: int = Field(ge=0, le=6)
    category: str
    payment_type: str
    discount_bucket: str
    price_bucket: str


class PredictionResponse(BaseModel):
    return_probability: float
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]
    recommended_action: Literal["INTERVENE", "MONITOR", "NO_ACTION"]
    intervention_threshold: float
    monitor_threshold: float
    expected_avoided_return_cost: float
    intervention_cost: float
    net_expected_value: float
    explanation: dict


class HealthResponse(BaseModel):
    status: str
    model: str
    threshold: float
    llm_mode: str
