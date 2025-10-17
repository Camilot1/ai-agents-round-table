"""
MCP server exposing a single tool for random number generation.
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="RandomNumberServer",
    instructions="Provides utilities for generating random integers.",
)


@mcp.tool()
async def generate_random_number(
    minimum: int = 0,
    maximum: int = 100,
) -> dict[str, Any]:
    """Generate a random integer within the inclusive range [minimum, maximum]."""
    if minimum > maximum:
        raise ValueError("minimum must be less than or equal to maximum.")
    return {
        "minimum": minimum,
        "maximum": maximum,
        "value": random.randint(minimum, maximum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP random number server.")
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
