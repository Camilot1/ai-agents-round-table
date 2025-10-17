"""MCP server exposing production planning tools."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="ProductionOperationsServer",
    instructions=(
        "Provides production planning helpers for routing orders through the "
        "factory and checking current workload."
    ),
)

MAX_PRODUCTION_CELLS = 5

PRODUCTION_ORDERS: dict[str, dict[str, Any]] = {
    "ORD001": {
        "order_id": "ORD001",
        "status": "in_progress",
        "scheduled_completion": "2025-10-20",
        "price": 125_000,
    },
    "ORD002": {
        "order_id": "ORD002",
        "status": "ready_for_dispatch",
        "scheduled_completion": "2025-10-10",
        "price": 8_000,
    },
    "ORD003": {
        "order_id": "ORD003",
        "status": "in_progress",
        "scheduled_completion": "2025-10-18",
        "price": 45_000,
    },
}

VALID_STATUSES: tuple[str, ...] = (
    "queued",
    "in_progress",
    "ready_for_dispatch",
    "completed",
    "cancelled",
)


@mcp.tool()
async def submit_order_to_production(
    order_id: str,
    price: float,
    desired_completion: str,
) -> dict[str, Any]:
    """Queue an order for production if capacity allows."""
    if price <= 0:
        raise ValueError("price must be greater than zero.")

    in_progress = sum(
        1 for order in PRODUCTION_ORDERS.values() if order["status"] == "in_progress"
    )
    available_capacity = MAX_PRODUCTION_CELLS - in_progress
    success = available_capacity > 0 and random.choice([True, False])

    if success:
        PRODUCTION_ORDERS[order_id] = {
            "order_id": order_id,
            "status": "queued",
            "scheduled_completion": desired_completion,
            "price": price,
        }
        message = (
            f"Order {order_id} scheduled for production; team notified to prepare."
        )
    else:
        message = (
            "Production cells are currently occupied. Order placed in standby queue."
        )

    return {
        "order_id": order_id,
        "success": success,
        "message": message,
        "current_load": in_progress,
        "max_capacity": MAX_PRODUCTION_CELLS,
        "estimated_start": (
            datetime.now() + timedelta(hours=4)
        ).isoformat(timespec="minutes"),
    }


@mcp.tool()
async def get_production_orders_by_status(status: str) -> list[dict[str, Any]]:
    """Return all production orders matching the provided status."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"status must be one of {', '.join(VALID_STATUSES)}."
        )
    return [
        order
        for order in PRODUCTION_ORDERS.values()
        if order["status"] == status
    ]


@mcp.tool()
async def get_production_order_by_id(order_id: str) -> dict[str, Any]:
    """Return a single production order entry by identifier."""
    order = PRODUCTION_ORDERS.get(order_id)
    if order is None:
        return {
            "order_id": order_id,
            "status": "not_found",
            "error": "Order is not registered in the current production queue.",
        }
    return order


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP production operations server.")
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
