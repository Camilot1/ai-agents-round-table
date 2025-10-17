"""
MCP server exposing a tool for retrieving the current time.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from typing import Any

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="TimeServer",
    instructions="Provides utilities for retrieving the current time in various zones.",
)


def _resolve_timezone(value: str) -> timezone:
    normalized = value.strip().lower()
    if normalized == "utc":
        return timezone.utc

    if normalized in {"local", "system"}:
        local_tz = datetime.now().astimezone().tzinfo
        return local_tz or timezone.utc

    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError:
        pass

    try:
        sign = 1
        if normalized.startswith("+"):
            normalized = normalized[1:]
        elif normalized.startswith("-"):
            normalized = normalized[1:]
            sign = -1

        if ":" in normalized:
            hours_str, minutes_str = normalized.split(":", maxsplit=1)
        else:
            hours_str, minutes_str = normalized[:2], normalized[2:]

        offset = sign * (int(hours_str) * 60 + int(minutes_str or 0))
        return timezone(timedelta(minutes=offset))
    except (ValueError, IndexError):
        raise ValueError(
            "Unsupported timezone value. Use 'UTC', 'local', or a numeric offset like '+03:00', "
            "or provide an IANA timezone name such as 'Europe/Moscow'."
        ) from None


@mcp.tool()
async def get_current_time(
    timezone_name: str = "UTC",
    output_format: str = "iso",
) -> dict[str, Any]:
    """Return the current time in the requested timezone."""
    tz = _resolve_timezone(timezone_name)
    now = datetime.now(tz=tz)

    if output_format == "iso":
        value: Any = now.isoformat()
    elif output_format == "epoch":
        value = now.timestamp()
    else:
        raise ValueError("output_format must be either 'iso' or 'epoch'.")

    return {
        "timezone": timezone_name,
        "format": output_format,
        "value": value,
        "utc_offset_minutes": int(now.utcoffset().total_seconds() // 60) if now.utcoffset() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP time server.")
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
