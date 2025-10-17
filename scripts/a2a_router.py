from __future__ import annotations

import argparse
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict

from python_a2a import A2AClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("a2a_router")


def load_agent_config(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"A2A agent config not found: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse {path}") from exc

    agents = data.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise RuntimeError("Agent config must define a non-empty 'agents' mapping of name -> url.")

    return {str(name): str(url) for name, url in agents.items()}


class RouterState:
    def __init__(self, agent_urls: Dict[str, str]) -> None:
        self.agent_urls = agent_urls
        self._clients: Dict[str, A2AClient] = {}

    def list_agents(self) -> Dict[str, str]:
        return self.agent_urls

    def get_client(self, agent: str) -> A2AClient:
        if agent not in self.agent_urls:
            raise KeyError(f"Unknown agent '{agent}'. Known agents: {', '.join(self.agent_urls)}")

        if agent not in self._clients:
            logger.info("Connecting to agent '%s' at %s", agent, self.agent_urls[agent])
            self._clients[agent] = A2AClient(self.agent_urls[agent])
        return self._clients[agent]


class A2ARouterHandler(BaseHTTPRequestHandler):
    state: RouterState  # will be injected before server starts

    def _set_json_headers(self, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - required by http.server
        if self.path.rstrip("/") == "/agents":
            self._set_json_headers()
            payload = {"agents": self.state.list_agents()}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return

        self._set_json_headers(HTTPStatus.NOT_FOUND)
        self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - required by http.server
        if self.path.rstrip("/") != "/relay":
            self._set_json_headers(HTTPStatus.NOT_FOUND)
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._set_json_headers(HTTPStatus.BAD_REQUEST)
            self.wfile.write(json.dumps({"error": "Invalid JSON payload"}).encode("utf-8"))
            return

        target = payload.get("target")
        message = payload.get("message")
        conversation_id = payload.get("conversation_id")

        if not isinstance(target, str) or not isinstance(message, str):
            self._set_json_headers(HTTPStatus.BAD_REQUEST)
            self.wfile.write(
                json.dumps({"error": "Payload must include 'target' and 'message' strings."}).encode("utf-8")
            )
            return

        try:
            client = self.state.get_client(target)
            response_text = client.ask(message, conversation_id=conversation_id)
        except Exception as exc:  # pragma: no cover - surfaces runtime issues
            logger.exception("Failed to relay message to %s: %s", target, exc)
            self._set_json_headers(HTTPStatus.INTERNAL_SERVER_ERROR)
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        self._set_json_headers()
        self.wfile.write(json.dumps({"response": response_text}).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - signature defined by BaseHTTPRequestHandler
        logger.info("%s - %s", self.address_string(), format % args)


def run_router(host: str, port: int, config_path: Path) -> None:
    agent_urls = load_agent_config(config_path)
    logger.info("Loaded %d agent definitions from %s", len(agent_urls), config_path)

    router_state = RouterState(agent_urls)
    handler_class = A2ARouterHandler
    handler_class.state = router_state  # type: ignore[attr-defined]

    server = ThreadingHTTPServer((host, port), handler_class)
    logger.info("A2A router listening on http://%s:%s", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting.")
    finally:
        server.server_close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple HTTP bridge that relays requests between A2A agents.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7080, help="Port to bind (default: 7080).")
    parser.add_argument(
        "--config",
        default="scripts/a2a_agents.json",
        help="Path to JSON config with agent name-to-URL mapping (default: scripts/a2a_agents.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_router(args.host, args.port, Path(args.config).resolve())
