"""MCP server exposing logistics coordination tools."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="LogisticsOperationsServer",
    instructions=(
        "Provides logistics helpers for tracking shipments, planning routes, "
        "and coordinating warehouse operations."
    ),
)

ROUTES: list[dict[str, Any]] = [
    {
        "route_id": "R001",
        "origin": "Moscow",
        "destination": "Saint Petersburg",
        "distance_km": 700,
        "duration_hours": 12,
    },
    {
        "route_id": "R002",
        "origin": "Moscow",
        "destination": "Kazan",
        "distance_km": 820,
        "duration_hours": 16,
    },
    {
        "route_id": "R003",
        "origin": "Saint Petersburg",
        "destination": "Nizhny Novgorod",
        "distance_km": 960,
        "duration_hours": 18,
    },
]

WAREHOUSES: dict[str, dict[str, Any]] = {
    "WH-MSK": {
        "name": "Moscow Central Hub",
        "capacity_pallets": 500,
        "occupied_pallets": 420,
        "next_receiving_slot": "2025-10-16T09:00",
    },
    "WH-SPB": {
        "name": "Saint Petersburg Distribution",
        "capacity_pallets": 350,
        "occupied_pallets": 185,
        "next_receiving_slot": "2025-10-15T14:00",
    },
    "WH-KZN": {
        "name": "Kazan Regional Hub",
        "capacity_pallets": 280,
        "occupied_pallets": 265,
        "next_receiving_slot": "2025-10-17T08:00",
    },
}

FLEET: dict[str, dict[str, Any]] = {
    "TRK-101": {
        "vehicle_id": "TRK-101",
        "type": "Truck",
        "capacity_kg": 8_000,
        "status": "available",
        "location": "Moscow",
    },
    "TRK-205": {
        "vehicle_id": "TRK-205",
        "type": "Refrigerated Truck",
        "capacity_kg": 5_500,
        "status": "in_transit",
        "location": "M11 Highway",
    },
    "VAN-044": {
        "vehicle_id": "VAN-044",
        "type": "Van",
        "capacity_kg": 1_500,
        "status": "available",
        "location": "Saint Petersburg",
    },
}

SHIPMENTS: dict[str, dict[str, Any]] = {
    "SHP001": {
        "order_id": "ORD210",
        "route_id": "R001",
        "status": "in_transit",
        "last_update": "2025-10-12T08:30",
        "eta": "2025-10-12T20:00",
    },
    "SHP002": {
        "order_id": "ORD225",
        "route_id": "R002",
        "status": "awaiting_pickup",
        "last_update": "2025-10-11T16:45",
        "eta": "2025-10-13T09:00",
    },
}

DRIVERS: list[str] = [
    "Sergey Ivanov",
    "Maria Petrova",
    "Alexey Smirnov",
    "Olga Kuznetsova",
]


#@mcp.tool() # Заглушка метода (отключена
async def get_shipment_status(shipment_id: str) -> dict[str, Any]:
    """Return current status information about a shipment."""
    shipment = SHIPMENTS.get(shipment_id)
    if shipment is None:
        status = random.choice(["in_transit", "delayed", "delivered"])
        eta = datetime.now() + timedelta(hours=random.randint(4, 24))
        return {
            "shipment_id": shipment_id,
            "status": status,
            "eta": eta.isoformat(timespec="minutes"),
            "message": "No cached data found; generated synthetic status.",
        }
    return {"shipment_id": shipment_id, **shipment}


#@mcp.tool() # Заглушка метода (отключена
async def list_available_routes(
    origin: str | None = None,
    destination: str | None = None,
) -> list[dict[str, Any]]:
    """List planned transport routes, optionally filtered by origin/destination."""
    def _matches(route: dict[str, Any]) -> bool:
        origin_ok = (
            origin is None
            or route["origin"].lower() == origin.lower()
        )
        destination_ok = (
            destination is None
            or route["destination"].lower() == destination.lower()
        )
        return origin_ok and destination_ok

    return [route for route in ROUTES if _matches(route)]


#@mcp.tool() # Заглушка метода (отключена)
async def calculate_delivery_eta(route_id: str) -> dict[str, Any]:
    """Estimate arrival time for a given route including randomised delays."""
    route = next((item for item in ROUTES if item["route_id"] == route_id), None)
    if route is None:
        raise ValueError(f"unknown route: {route_id}")

    base_eta = datetime.now() + timedelta(hours=route["duration_hours"])
    delay_hours = random.choice([0, 1, 2, 3])
    eta = base_eta + timedelta(hours=delay_hours)
    return {
        "route_id": route_id,
        "planned_duration_hours": route["duration_hours"],
        "delay_hours": delay_hours,
        "estimated_arrival": eta.isoformat(timespec="minutes"),
    }


@mcp.tool() # Заглушка метода
async def calculate_shipping_cost(
    weight_kg: float,
    distance_km: float,
    priority: Literal["standard", "express", "overnight"] = "standard",
) -> dict[str, Any]:
    """Calculate a rough transport cost estimate based on weight and distance."""
    if weight_kg <= 0:
        raise ValueError("weight_kg must be greater than zero.")
    if distance_km <= 0:
        raise ValueError("distance_km must be greater than zero.")

    priority_multiplier = {
        "standard": 1.0,
        "express": 1.35,
        "overnight": 1.85,
    }[priority]

    base_rate = 2.5  # currency units per km per 100 kg
    cost = base_rate * (distance_km) * (weight_kg / 100)
    total_cost = round(cost * priority_multiplier, 2)
    return {
        "estimated_cost": total_cost,
        "currency": "RUB",
    }


#@mcp.tool() # Заглушка метода (отключена
async def schedule_pickup(
    order_id: str,
    warehouse_id: str,
    pickup_window_start: str,
    pickup_window_end: str,
) -> dict[str, Any]:
    """Reserve a pickup window at the specified warehouse."""
    warehouse = WAREHOUSES.get(warehouse_id)
    if warehouse is None:
        raise ValueError(f"unknown warehouse: {warehouse_id}")

    driver = random.choice(DRIVERS)
    vehicle = random.choice([vehicle for vehicle in FLEET.values()])
    confirmation = f"PU-{random.randint(1000, 9999)}"
    return {
        "order_id": order_id,
        "warehouse_id": warehouse_id,
        "warehouse_name": warehouse["name"],
        "assigned_driver": driver,
        "vehicle_id": vehicle["vehicle_id"],
        "pickup_window_start": pickup_window_start,
        "pickup_window_end": pickup_window_end,
        "confirmation_code": confirmation,
    }


#@mcp.tool() # Заглушка метода (отключена
async def assign_vehicle(route_id: str) -> dict[str, Any]:
    """Allocate an available vehicle to the route."""
    available = [
        vehicle for vehicle in FLEET.values() if vehicle["status"] == "available"
    ]
    if not available:
        raise RuntimeError("no vehicles are currently available.")

    vehicle = random.choice(available)
    return {
        "route_id": route_id,
        "vehicle_id": vehicle["vehicle_id"],
        "capacity_kg": vehicle["capacity_kg"],
        "status": "assigned",
        "dispatch_time": (datetime.now() + timedelta(hours=1)).isoformat(
            timespec="minutes"
        ),
    }


#@mcp.tool() # Заглушка метода (отключена
async def get_warehouse_status(warehouse_id: str) -> dict[str, Any]:
    """Return storage utilisation data for the requested warehouse."""
    warehouse = WAREHOUSES.get(warehouse_id)
    if warehouse is None:
        raise ValueError(f"unknown warehouse: {warehouse_id}")

    utilisation = warehouse["occupied_pallets"] / warehouse["capacity_pallets"]
    return {
        "warehouse_id": warehouse_id,
        "name": warehouse["name"],
        "capacity_pallets": warehouse["capacity_pallets"],
        "occupied_pallets": warehouse["occupied_pallets"],
        "utilisation_percent": round(utilisation * 100, 1),
        "next_receiving_slot": warehouse["next_receiving_slot"],
    }


#@mcp.tool() # Заглушка метода (отключена
async def track_vehicle(vehicle_id: str) -> dict[str, Any]:
    """Provide location and status for a vehicle in the fleet."""
    vehicle = FLEET.get(vehicle_id)
    if vehicle is None:
        raise ValueError(f"unknown vehicle: {vehicle_id}")

    return {
        "vehicle_id": vehicle_id,
        "status": vehicle["status"],
        "location": vehicle["location"],
        "last_update": datetime.now().isoformat(timespec="minutes"),
        "temperature_c": round(random.uniform(4.0, 18.0), 1),
    }


#@mcp.tool() # Заглушка метода (отключена
async def report_delivery_issue(
    shipment_id: str,
    issue: str,
    severity: Literal["low", "medium", "high"] = "medium",
) -> dict[str, Any]:
    """Log a delivery issue and return an acknowledgement record."""
    ticket_id = f"ISS-{random.randint(1000, 9999)}"
    return {
        "ticket_id": ticket_id,
        "shipment_id": shipment_id,
        "severity": severity,
        "issue": issue,
        "created_at": datetime.now().isoformat(timespec="minutes"),
        "status": "acknowledged",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP logistics operations server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
        help="Transport protocol to use (defaults to stdio).",
    )
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
