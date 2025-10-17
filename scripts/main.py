from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

from logging.handlers import RotatingFileHandler

from mcp import StdioServerParameters

from a2a_agent import A2AAgentConfig, run_a2a_server
from telegram_bot import (
    A2APeerClient,
    BotConfig,
    LLMAgent,
    MCPServerEndpoint,
    MCPToolClient,
    TelegramBot,
)

logger = logging.getLogger(__name__)


def _clear_existing_log_files(log_path: Path) -> None:
    """Remove previous log file and any rotated variants before new run."""
    candidates = [log_path]
    try:
        candidates.extend(
            entry
            for entry in log_path.parent.glob(f"{log_path.name}.*")
            if entry.is_file()
        )
    except OSError as exc:  # pragma: no cover - filesystem issues are rare
        logger.warning("Failed to enumerate rotated logs for %s: %s", log_path, exc)
        candidates = [log_path]

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except OSError as exc:
            logger.warning("Failed to remove old log file %s: %s", candidate, exc)


@dataclass(slots=True)
class MCPServerSettings:
    name: str
    command: List[str]
    cwd: Optional[Path]
    env: Optional[Dict[str, str]]


@dataclass(slots=True)
class Settings:
    telegram_token: str
    allowed_users: Optional[Set[int]]
    hf_token: Optional[str]
    hf_provider: Optional[str]
    model_name: str
    local_model_path: Optional[str]
    mcp_servers: List[MCPServerSettings]
    a2a_peers: List["A2APeer"]
    a2a_enabled: bool
    a2a_host: str
    a2a_port: int
    a2a_name: str
    a2a_description: str
    a2a_version: str
    a2a_public_url: str
    temperature: float = 0.2
    max_output_tokens: int = 512


@dataclass(slots=True)
class A2APeer:
    name: str
    url: str


def _parse_allowed_users(raw: str) -> Optional[Set[int]]:
    raw = raw.strip()
    if not raw:
        return None

    users: Set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            users.add(int(item))
        except ValueError as exc:
            raise ValueError(f"ALLOWED_USERS contains a non-numeric value: {item}") from exc
    return users


def _resolve_command_part(project_root: Path, part: str) -> str:
    if part == "{python}":
        return sys.executable

    candidate_path = Path(part)
    if candidate_path.is_absolute():
        return str(candidate_path)

    potential = (project_root / candidate_path).resolve()
    if potential.exists():
        return str(potential)

    return part


def _resolve_optional_path(project_root: Path, value: str) -> Path:
    path_value = Path(value)
    return path_value if path_value.is_absolute() else (project_root / path_value).resolve()


def _load_mcp_servers(project_root: Path, config_path: Path) -> List[MCPServerSettings]:
    if not config_path.exists():
        raise RuntimeError(f"MCP servers config file not found: {config_path}")

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse MCP servers config: {config_path}") from exc

    servers_data = data.get("servers")
    if not isinstance(servers_data, list) or not servers_data:
        raise RuntimeError("MCP servers config must define a non-empty 'servers' list.")

    servers: List[MCPServerSettings] = []
    for entry in servers_data:
        if not isinstance(entry, dict):
            raise ValueError("Each MCP server entry must be a JSON object.")

        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each MCP server entry must include a non-empty 'name'.")
        name = name.strip()

        command = entry.get("command")
        script = entry.get("script")

        if command:
            if not isinstance(command, list) or not command:
                raise ValueError(f"MCP server '{name}' has invalid 'command' definition.")
            command_parts = [str(part) for part in command]
        elif script:
            if not isinstance(script, str) or not script.strip():
                raise ValueError(f"MCP server '{name}' has invalid 'script' value.")
            script_path = script.strip()
            python_exec = entry.get("python", "{python}")
            extra_args = entry.get("args", [])
            if not isinstance(extra_args, list):
                raise ValueError(f"MCP server '{name}' has invalid 'args' definition.")
            command_parts = [str(python_exec), script_path, *map(str, extra_args)]
        else:
            raise ValueError(f"MCP server '{name}' must define either 'command' or 'script'.")

        resolved_command = [
            _resolve_command_part(project_root, part) for part in command_parts
        ]
        if not resolved_command:
            raise ValueError(f"MCP server '{name}' resulted in an empty command.")

        cwd_value = entry.get("cwd")
        cwd_path = _resolve_optional_path(project_root, cwd_value) if cwd_value else None

        env_value = entry.get("env")
        env: Optional[Dict[str, str]] = None
        if env_value is not None:
            if not isinstance(env_value, dict):
                raise ValueError(f"MCP server '{name}' has invalid 'env' definition.")
            env = {str(k): str(v) for k, v in env_value.items()}

        servers.append(
            MCPServerSettings(
                name=name,
                command=resolved_command,
                cwd=cwd_path,
                env=env,
            )
        )

    return servers


def _load_a2a_peers(config_path: Path, self_name: str) -> List[A2APeer]:
    if not config_path.exists():
        logger.info("A2A peers config %s not found; continuing without peers.", config_path)
        return []

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse A2A peers config: {config_path}") from exc

    raw_peers = data.get("agents", data)
    if not isinstance(raw_peers, dict):
        raise RuntimeError("A2A peers config must be a mapping of agent name to URL.")

    peers: List[A2APeer] = []
    for name, url in raw_peers.items():
        if not isinstance(name, str) or not isinstance(url, str):
            raise ValueError("Agent names and URLs must be strings.")
        if name.strip() and name.strip() != self_name:
            peers.append(A2APeer(name=name.strip(), url=url.strip()))
    return peers


def load_settings(env_filename: str = ".env") -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / env_filename

    if load_dotenv is not None and env_file.exists():
        load_dotenv(env_file, override=False)

    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

    allowed_users = _parse_allowed_users(os.getenv("ALLOWED_USERS", ""))
    hf_token = os.getenv("HF_TOKEN")
    hf_provider = os.getenv("HF_INFERENCE_PROVIDER", "hf-inference").strip() or "hf-inference"
    model_name = os.getenv("HF_MODEL_ID", "google/gemma-3-1b-it")
    local_model_raw = os.getenv("LOCAL_MODEL_PATH", "").strip()
    local_model_path: Optional[str] = None
    if local_model_raw:
        local_model_path = str(_resolve_optional_path(project_root, local_model_raw))

    mcp_config_name = os.getenv("MCP_SERVERS_CONFIG", "scripts/mcp_servers.json")
    mcp_config_path = project_root / mcp_config_name
    mcp_servers = _load_mcp_servers(project_root, mcp_config_path)

    a2a_enabled = os.getenv("A2A_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    a2a_host = os.getenv("A2A_HOST", "127.0.0.1")
    a2a_port = int(os.getenv("A2A_PORT", "7000"))
    a2a_name = os.getenv("A2A_NAME", "Gemma MCP Agent")
    a2a_description = os.getenv(
        "A2A_DESCRIPTION",
        "LLM assistant powered by Gemma and MCP tools.",
    )
    a2a_version = os.getenv("A2A_VERSION", "1.0.0")
    a2a_public_url = os.getenv("A2A_PUBLIC_URL", f"http://localhost:{a2a_port}")
    peers_config_name = os.getenv("A2A_PEERS_CONFIG", "scripts/a2a_agents.json")
    peers_config_path = Path(peers_config_name)
    if not peers_config_path.is_absolute():
        peers_config_path = (project_root / peers_config_path).resolve()

    return Settings(
        telegram_token=telegram_token,
        allowed_users=allowed_users,
        hf_token=hf_token,
        hf_provider=hf_provider,
        model_name=model_name,
        local_model_path=local_model_path,
        mcp_servers=mcp_servers,
        a2a_peers=_load_a2a_peers(peers_config_path, a2a_name),
        a2a_enabled=a2a_enabled,
        a2a_host=a2a_host,
        a2a_port=a2a_port,
        a2a_name=a2a_name,
        a2a_description=a2a_description,
        a2a_version=a2a_version,
        a2a_public_url=a2a_public_url,
    )


def build_mcp_endpoints(settings: Settings) -> List[MCPServerEndpoint]:
    if not settings.mcp_servers:
        raise RuntimeError("No MCP servers configured.")
    return [
        MCPServerEndpoint(
            name=server.name,
            parameters=StdioServerParameters(
                command=server.command[0],
                args=server.command[1:],
                cwd=str(server.cwd) if server.cwd else None,
                env=server.env,
            ),
        )
        for server in settings.mcp_servers
    ]


async def run_bot(settings: Settings) -> None:
    logger.info("run_bot: building MCP endpoints.")
    endpoints = build_mcp_endpoints(settings)
    logger.info("run_bot: %d MCP endpoints configured.", len(endpoints))

    tool_client = MCPToolClient(endpoints)
    peer_client = A2APeerClient(settings.a2a_peers)
    logger.info("run_bot: initialising LLMAgent.")

    agent = LLMAgent(
        model_name=settings.model_name,
        hf_token=settings.hf_token,
        inference_provider=settings.hf_provider,
        local_model_path=settings.local_model_path,
        tool_client=tool_client,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        peer_client=peer_client,
        instance_label="telegram",
    )
    logger.info("run_bot: LLMAgent initialised.")

    bot_config = BotConfig(
        token=settings.telegram_token,
        allowed_users=settings.allowed_users,
    )
    bot = TelegramBot(bot_config, agent)
    logger.info("run_bot: TelegramBot constructed; entering polling loop.")

    try:
        await bot.run()
    finally:
        await tool_client.close()
        peer_client.close()
        logger.info("run_bot: shutdown complete.")


def run_a2a(settings: Settings) -> None:
    if not settings.a2a_enabled:
        raise RuntimeError("A2A mode is disabled in configuration.")

    logger.info("run_a2a: building MCP endpoints.")
    endpoints = build_mcp_endpoints(settings)
    logger.info("run_a2a: %d MCP endpoints configured.", len(endpoints))
    tool_client = MCPToolClient(endpoints)
    peer_client = A2APeerClient(settings.a2a_peers)
    logger.info("run_a2a: initialising LLMAgent.")

    llm_agent = LLMAgent(
        model_name=settings.model_name,
        hf_token=settings.hf_token,
        inference_provider=settings.hf_provider,
        local_model_path=settings.local_model_path,
        tool_client=tool_client,
        temperature=settings.temperature,
        max_output_tokens=settings.max_output_tokens,
        peer_client=peer_client,
        instance_label="a2a",
    )
    logger.info("run_a2a: LLMAgent initialised.")

    config = A2AAgentConfig(
        name=settings.a2a_name,
        description=settings.a2a_description,
        version=settings.a2a_version,
        url=settings.a2a_public_url,
    )
    logger.info("run_a2a: launching server thread.")

    try:
        run_a2a_server(
            llm_agent=llm_agent,
            config=config,
            host=settings.a2a_host,
            port=settings.a2a_port,
        )
    finally:
        peer_client.close()
        asyncio.run(tool_client.close())
        logger.info("run_a2a: shutdown complete.")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("aiogram").setLevel(logging.INFO)


def configure_file_logging(log_name: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    override = os.getenv("LOG_FILE")

    if override:
        log_path = Path(override).expanduser()
        if not log_path.is_absolute():
            log_path = (project_root / log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", log_name).strip("_") or "agent"
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{safe_name}.log"

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_is_codex_file_handler", False):
            return log_path

    _clear_existing_log_files(log_path)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    file_handler._is_codex_file_handler = True  # type: ignore[attr-defined]
    root_logger.addHandler(file_handler)
    logging.info("File logging enabled at %s", log_path)
    return log_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Gemma+MCP assistant with optional Telegram and A2A frontends."
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to the environment file relative to project root (default: .env).",
    )
    parser.add_argument(
        "--mode",
        choices=("telegram", "a2a", "both"),
        default="telegram",
        help="Select between the Telegram bot, the A2A server, or both (default: telegram).",
    )
    parser.add_argument(
        "--a2a-host",
        help="Override A2A server host (falls back to A2A_HOST env).",
    )
    parser.add_argument(
        "--a2a-port",
        type=int,
        help="Override A2A server port (falls back to A2A_PORT env).",
    )
    parser.add_argument(
        "--a2a-name",
        help="Override A2A agent name.",
    )
    parser.add_argument(
        "--a2a-description",
        help="Override A2A agent description.",
    )
    parser.add_argument(
        "--a2a-version",
        help="Override A2A agent version string.",
    )
    parser.add_argument(
        "--a2a-url",
        help="Override the publicly advertised URL for the A2A agent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    settings = load_settings(args.env)

    if args.a2a_host:
        settings.a2a_host = args.a2a_host
    if args.a2a_port:
        settings.a2a_port = args.a2a_port
    if args.a2a_name:
        settings.a2a_name = args.a2a_name
    if args.a2a_description:
        settings.a2a_description = args.a2a_description
    if args.a2a_version:
        settings.a2a_version = args.a2a_version
    if args.a2a_url:
        settings.a2a_public_url = args.a2a_url

    configure_file_logging(settings.a2a_name or "agent")

    if args.mode in {"a2a", "both"}:
        settings.a2a_enabled = True

    if args.mode == "telegram":
        asyncio.run(run_bot(settings))
        return

    if args.mode == "a2a":
        run_a2a(settings)
        return

    if args.mode == "both":
        logger.info(
            "Starting both Telegram bot and A2A server (A2A on %s:%s)",
            settings.a2a_host,
            settings.a2a_port,
        )
        a2a_thread = threading.Thread(target=run_a2a, args=(settings,), daemon=True)
        a2a_thread.start()
        try:
            asyncio.run(run_bot(settings))
        finally:
            logger.info("Telegram bot stopped; main process exiting.")


if __name__ == "__main__":
    main()
