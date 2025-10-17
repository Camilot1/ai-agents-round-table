"""MCP server exposing sales team utilities as tools."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="SalesOperationsServer",
    instructions=(
        "Provides sales support helpers for working with customer profiles, "
        "product availability, and order lifecycle management."
    ),
)

PRODUCT_CATALOG: list[dict[str, Any]] = [
    {"name": "Laser Cutter L-100", "sku": "LC-L100", "price": 125_000},
    {"name": "Packaging Line P-20", "sku": "PL-P20", "price": 8_000},
    {"name": "Quality Scanner QS-5", "sku": "QS-5", "price": 3_000},
    {"name": "Warehouse Drone WD-4", "sku": "WD-4", "price": 15_000},
]

MANAGERS: list[dict[str, Any]] = [
    {
        "manager_id": "M001",
        "name": "Anna Sergeeva",
        "products": ["Laser Cutter L-100", "Warehouse Drone WD-4"],
    },
    {
        "manager_id": "M002",
        "name": "Dmitry Volkov",
        "products": ["Packaging Line P-20", "Quality Scanner QS-5"],
    },
    {
        "manager_id": "M003",
        "name": "Irina Pavlova",
        "products": ["Laser Cutter L-100", "Packaging Line P-20"],
    },
]

MEETING_SLOTS: list[dict[str, Any]] = [
    {
        "manager": "Anna Sergeeva",
        "product": "Laser Cutter L-100",
        "available_slots": ["2025-10-15", "2025-10-17"],
    },
    {
        "manager": "Dmitry Volkov",
        "product": "Packaging Line P-20",
        "available_slots": ["2025-10-16", "2025-10-18"],
    },
    {
        "manager": "Irina Pavlova",
        "product": "Quality Scanner QS-5",
        "available_slots": ["2025-10-14", "2025-10-20"],
    },
]

LOYALTY_OFFERS: list[str] = [
    "Seasonal discount: 10% off any follow-up order placed this month.",
    "Extended warranty upgrade available for orders confirmed before Friday.",
    "Bundle promotion: add a Packaging Line P-20 and save 25% on installation.",
]

ORDER_STATUSES: tuple[str, ...] = (
    "new",
    "in_production",
    "ready_for_shipment",
    "delivered",
)

ORDER_PROGRESS = {
    "new": 10,
    "in_production": 55,
    "ready_for_shipment": 85,
    "delivered": 100,
}


#@mcp.tool()
async def get_customer_profile(client_id: str) -> dict[str, Any]:
    """Return a synthetic customer profile with current active orders."""
    return {
        "client_id": client_id,
        "name": "Ivan Ivanov",
        "phone": "+7-900-555-12-34",
        "email": "ivanov@example.com",
        "active_orders": [
            {"order_id": "ORD001", "status": "in_production"},
            {"order_id": "ORD002", "status": "ready_for_shipment"},
        ],
    }


@mcp.tool()
async def get_available_products() -> list[dict[str, Any]]:
    """List catalogue items that can be offered to customers."""
    return PRODUCT_CATALOG


#@mcp.tool()
async def get_order_status(order_id: str) -> dict[str, Any]:
    """Return a simulated production status with progress for the order."""
    status = random.choice(ORDER_STATUSES)
    manager = random.choice(MANAGERS)
    eta = datetime.now() + timedelta(days=random.randint(1, 5))
    return {
        "order_id": order_id,
        "status": status,
        "progress_percent": ORDER_PROGRESS[status],
        "expected_completion": eta.strftime("%Y-%m-%d"),
        "manager": manager["name"],
    }


#@mcp.tool()
async def get_order_history(client_id: str) -> dict[str, Any]:
    """Return recent order history for the requested client."""
    history = [
        {"order_id": "ORD000", "date": "2025-09-01", "status": "delivered"},
        {"order_id": "ORD001", "date": "2025-09-15", "status": "ready_for_shipment"},
        {"order_id": "ORD002", "date": "2025-10-01", "status": "in_production"},
    ]
    return {"client_id": client_id, "orders": history}


@mcp.tool()
async def calculate_order_price(
    product_name: str,
    quantity: int,
    client_id: str,
) -> dict[str, Any]:
    """Estimate an order cost including a randomised promotional discount."""
    if quantity < 1:
        raise ValueError("quantity must be a positive integer.")

    product = next(
        (item for item in PRODUCT_CATALOG if item["name"] == product_name), None
    )
    if product is None:
        raise ValueError(f"unknown product: {product_name}")

    discount = random.choice([0.0, 0.05, 0.1])
    total = product["price"] * quantity * (1 - discount)
    return {
        "client_id": client_id,
        "product_name": product_name,
        "quantity": quantity,
        "unit_price": product["price"],
        "discount": f"{int(discount * 100)}%",
        "total_price": round(total, 2),
    }


@mcp.tool()
async def get_available_slots() -> list[dict[str, Any]]:
    """List free demo slots for product consultations."""
    return MEETING_SLOTS


@mcp.tool()
async def get_managers_info() -> list[dict[str, Any]]:
    """Return current sales managers and their specialisations."""
    return MANAGERS


#@mcp.tool()
async def manage_order(
    action: Literal["get", "create", "cancel"],
    order_id: str | None = None,
    product_id: str | None = None,
) -> dict[str, Any]:
    """Perform simple order management actions such as lookup, create, or cancel."""
    if action == "get":
        if not order_id:
            raise ValueError("order_id is required when action is 'get'.")
        return await get_order_status(order_id)

    if action == "create":
        if not product_id:
            raise ValueError("product_id is required when action is 'create'.")
        manager = random.choice(MANAGERS)
        return {
            "order_id": f"ORD{random.randint(100, 999)}",
            "product_id": product_id,
            "assigned_manager": manager["name"],
            "status": "new",
            "message": "Order created and assigned for follow-up.",
            "estimated_completion": (
                datetime.now() + timedelta(days=3)
            ).strftime("%Y-%m-%d"),
        }

    if action == "cancel":
        if not order_id:
            raise ValueError("order_id is required when action is 'cancel'.")
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "Order flagged for cancellation and customer notified.",
        }

    raise ValueError(f"unsupported action: {action}")


#@mcp.tool()
async def notify_customer(client_id: str) -> dict[str, Any]:
    """Suggest a loyalty offer that can be sent to the customer."""
    return {
        "client_id": client_id,
        "offer": random.choice(LOYALTY_OFFERS),
        "valid_until": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP sales operations server.")
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
