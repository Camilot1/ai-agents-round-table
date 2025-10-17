from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from aiogram import Bot, Dispatcher
from aiogram.enums import ChatAction, ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.exceptions import TelegramBadRequest
from huggingface_hub import AsyncInferenceClient
from huggingface_hub.errors import RepositoryNotFoundError
from mcp import ClientSession, StdioServerParameters, Tool, stdio_client
from mcp import types as mcp_types
from python_a2a.client.http import A2AClient
from python_a2a.exceptions import A2AError

try:  # aiogram >= 3.5
    from aiogram.client.default import DefaultBotProperties
except ImportError:  # pragma: no cover - older aiogram fallback
    DefaultBotProperties = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MCPServerEndpoint:
    name: str
    parameters: StdioServerParameters


@dataclass(slots=True)
class _ToolBinding:
    identifier: str
    server: str
    tool: Tool
    original_name: str


class MCPToolClient:
    """Manage persistent STDIO connections to one or more MCP servers."""

    def __init__(self, servers: List[MCPServerEndpoint]) -> None:
        if not servers:
            raise ValueError("At least one MCP server endpoint must be provided.")

        self._servers = servers
        self._stdio_contexts: Dict[str, Any] = {}
        self._session_contexts: Dict[str, Any] = {}
        self._sessions: Dict[str, ClientSession] = {}
        self._tools: Dict[str, _ToolBinding] = {}
        self._init_lock = asyncio.Lock()
        self._call_lock = asyncio.Lock()

    async def ensure_ready(self) -> None:
        if self._sessions:
            return

        async with self._init_lock:
            if self._sessions:
                return

            tool_map: Dict[str, _ToolBinding] = {}
            stdio_contexts: Dict[str, Any] = {}
            session_contexts: Dict[str, Any] = {}
            sessions: Dict[str, ClientSession] = {}

            try:
                for endpoint in self._servers:
                    name = endpoint.name
                    stdio_cm = stdio_client(endpoint.parameters)
                    read, write = await stdio_cm.__aenter__()

                    session_cm = ClientSession(read, write)
                    session = await session_cm.__aenter__()
                    await session.initialize()

                    stdio_contexts[name] = stdio_cm
                    session_contexts[name] = session_cm
                    sessions[name] = session

                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        identifier = self._build_tool_identifier(name, tool.name)
                        if identifier in tool_map:
                            raise ValueError(
                                f"Duplicate tool identifier '{identifier}' across MCP servers."
                            )
                        tool_map[identifier] = _ToolBinding(
                            identifier=identifier,
                            server=name,
                            tool=tool,
                            original_name=tool.name,
                        )
            except Exception:
                await self._cleanup_contexts(session_contexts, stdio_contexts)
                raise

            self._sessions = sessions
            self._session_contexts = session_contexts
            self._stdio_contexts = stdio_contexts
            self._tools = tool_map

    async def close(self) -> None:
        async with self._init_lock:
            await self._cleanup_contexts(self._session_contexts, self._stdio_contexts)
            self._sessions.clear()
            self._session_contexts.clear()
            self._stdio_contexts.clear()
            self._tools.clear()

    async def get_tool_descriptions(self) -> List[str]:
        await self.ensure_ready()
        descriptions: List[str] = []
        for binding in self._tools.values():
            tool = binding.tool
            schema = tool.inputSchema or {}
            properties = schema.get("properties", {})

            if properties:
                args_details = ", ".join(
                    f"{name}: {details.get('type', 'unknown')}"
                    for name, details in properties.items()
                )
                description = (
                    f"{binding.identifier}: {tool.description or 'No description provided.'} "
                    f"(arguments: {args_details})"
                )
            else:
                description = (
                    f"{binding.identifier}: {tool.description or 'No description provided.'}"
                )

            descriptions.append(description)

        return descriptions

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        logger.info("Peer tool invocation requested: %s with args %s", tool_name, arguments)
        async with self._call_lock:
            await self.ensure_ready()

            binding = self._tools.get(tool_name)
            if binding is None:
                alias_binding = self._resolve_tool_alias(tool_name)
                if alias_binding:
                    logger.debug(
                        "Resolved tool alias '%s' to registered identifier '%s'",
                        tool_name,
                        alias_binding.identifier,
                    )
                    binding = alias_binding
                    tool_name = alias_binding.identifier

            if binding is None:
                raise ValueError(f"Tool '{tool_name}' is not registered with the MCP client.")

            try:
                logger.debug("Calling MCP tool %s.%s with args %s", binding.server, binding.original_name, arguments)
                session = self._sessions[binding.server]
                result = await session.call_tool(binding.original_name, arguments)
                logger.debug("MCP tool %s.%s returned %s", binding.server, binding.original_name, result)
            except Exception:
                await self.close()
                raise

            return self._format_tool_result(result)

    @staticmethod
    def _format_tool_result(result: mcp_types.CallToolResult) -> str:
        parts: List[str] = []

        if getattr(result, "isError", False):
            parts.append("Tool reported an error:")

        for content in getattr(result, "content", []) or []:
            if isinstance(content, mcp_types.TextContent):
                parts.append(content.text)
            elif isinstance(content, mcp_types.EmbeddedResource):
                parts.append(f"[Embedded resource] {content.resource.uri}")
            else:
                parts.append(str(content))

        structured = getattr(result, "structuredContent", None)
        if structured:
            parts.append(json.dumps(structured, ensure_ascii=False, indent=2))

        return "\n".join(parts).strip() or "Tool executed without textual output."

    @staticmethod
    def _build_tool_identifier(server_name: str, tool_name: str) -> str:
        return f"{server_name}.{tool_name}"

    def _resolve_tool_alias(self, tool_name: str | None) -> _ToolBinding | None:
        if not tool_name:
            return None

        lowered = tool_name.lower()
        for binding in self._tools.values():
            if binding.identifier.lower() == lowered:
                return binding

        if "." not in tool_name:
            # Try exact suffix match first
            suffix_matches = [
                binding
                for binding in self._tools.values()
                if binding.identifier.split(".", 1)[-1] == tool_name
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]

            lowered_suffix_matches = [
                binding
                for binding in self._tools.values()
                if binding.identifier.split(".", 1)[-1].lower() == lowered
            ]
            if len(lowered_suffix_matches) == 1:
                return lowered_suffix_matches[0]

        return None

    @staticmethod
    async def _cleanup_contexts(
        session_contexts: Dict[str, Any],
        stdio_contexts: Dict[str, Any],
    ) -> None:
        for session_cm in session_contexts.values():
            try:
                await session_cm.__aexit__(None, None, None)
            except Exception:  # pragma: no cover - best effort cleanup
                logging.exception("Failed to close MCP session context", exc_info=True)

        for stdio_cm in stdio_contexts.values():
            try:
                await stdio_cm.__aexit__(None, None, None)
            except Exception:  # pragma: no cover - best effort cleanup
                logging.exception("Failed to close MCP stdio context", exc_info=True)


class LLMAgent:
    """LLM agent that can decide when to call MCP tools."""

    def __init__(
        self,
        *,
        model_name: str,
        hf_token: Optional[str],
        inference_provider: Optional[str],
        local_model_path: Optional[str],
        tool_client: MCPToolClient,
        peer_client: Optional[A2APeerClient] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 512,
        instance_label: str = "default",
    ) -> None:
        self._tool_client = tool_client
        self._peer_client = peer_client
        self._model_name = model_name
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._inference_provider = (inference_provider or "").strip() or None
        self._history: Dict[str, List[Dict[str, str]]] = {}
        self._system_prompt: Optional[str] = None
        self._max_tool_iterations = 2
        self._client: Optional[AsyncInferenceClient] = None
        self._llama = None
        self._backend = "hf"
        self._local_model_path: Optional[str] = None
        self._local_chat_format: Optional[str] = None
        self._instance_label = instance_label.strip()

        label_prefix = f"[{self._instance_label}] " if self._instance_label else ""

        resolved_local_path: Optional[Path] = None
        candidate_paths: List[str] = []
        if local_model_path:
            candidate_paths.append(local_model_path)

        model_lower = model_name.lower()
        if not local_model_path and model_lower.endswith(".gguf"):
            candidate_paths.append(model_name)

        for candidate in candidate_paths:
            candidate_path = Path(candidate).expanduser()
            if candidate_path.is_file():
                resolved_local_path = candidate_path.resolve()
                break

        if local_model_path and resolved_local_path is None:
            raise RuntimeError(f"LOCAL_MODEL_PATH points to '{local_model_path}', but the file does not exist.")

        if resolved_local_path is not None:
            try:
                from llama_cpp import Llama  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Using LOCAL_MODEL_PATH requires the 'llama-cpp-python' package. "
                    "Install it with 'pip install llama-cpp-python'."
                ) from exc

            self._backend = "local"
            self._local_model_path = str(resolved_local_path)
            self._model_name = resolved_local_path.name

            llama_kwargs: Dict[str, Any] = {
                "model_path": self._local_model_path,
                "n_ctx": 8192,
                "logits_all": False,
                "use_mlock": False,
                "verbose": False,
            }
            if "gemma" in self._model_name.lower():
                llama_kwargs["chat_format"] = "gemma"
                self._local_chat_format = "gemma"

            logger.info(
                "%sLoading local GGUF model from %s (chat_format=%s).",
                label_prefix,
                self._local_model_path,
                llama_kwargs.get("chat_format", "auto"),
            )
            load_start = time.perf_counter()
            load_result: Dict[str, Any] = {}

            def _load_model() -> None:
                try:
                    load_result["model"] = Llama(**llama_kwargs)
                except Exception as exc:  # pragma: no cover - surfaces runtime issues
                    load_result["error"] = exc

            loader = threading.Thread(target=_load_model, daemon=True)
            loader.start()

            while loader.is_alive():
                loader.join(timeout=5.0)
                elapsed = time.perf_counter() - load_start
                if loader.is_alive():
                    logger.info(
                        "%sStill loading local model... %.1f seconds elapsed.",
                        label_prefix,
                        elapsed,
                    )

            if "error" in load_result:
                raise load_result["error"]  # noqa: B904 - re-raise original error

            self._llama = load_result.get("model")
            load_duration = time.perf_counter() - load_start
            logger.info(
                "%sLocal model initialised from %s in %.2f seconds.",
                label_prefix,
                self._local_model_path,
                load_duration,
            )
        elif model_lower.endswith(".gguf"):
            raise RuntimeError(
                f"Model '{model_name}' appears to reference a GGUF file, but it was not found locally. "
                "Set LOCAL_MODEL_PATH to the absolute path of the GGUF model."
            )
        else:
            self._client = AsyncInferenceClient(
                model=model_name,
                token=hf_token,
                provider=self._inference_provider,
            )
            logger.info(
                "%sRemote inference client initialised for model %s.",
                label_prefix,
                model_name,
            )

        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

    async def ensure_ready(self) -> None:
        if self._system_prompt is not None:
            return

        await self._tool_client.ensure_ready()
        tool_descriptions = await self._tool_client.get_tool_descriptions()
        peer_descriptions: List[str] = []
        if self._peer_client and self._peer_client.has_peers():
            peer_descriptions = self._peer_client.get_tool_descriptions()
            tool_descriptions.extend(peer_descriptions)

        if not tool_descriptions:
            logger.warning(
                "No MCP or peer tools detected for model %s. Check MCP server and A2A configuration.",
                self._model_name,
            )

        logger.info("System prompt initialised for model %s with %d MCP tools and %d peer tools.", self._model_name, len(tool_descriptions) - len(peer_descriptions), len(peer_descriptions))
        tools_text = "\n".join(f"- {item}" for item in tool_descriptions)
        self._system_prompt = (
            "You are a helpful assistant that can call external tools when necessary.\n"
            "Available tools:\n"
            f"{tools_text}\n"
            "Remember:\n"
            "- Treat every item above as available for the entire conversation.\n"
            "- Entries starting with 'agent.' let you talk to external agents via the A2A protocol; when the user asks for another department's help, call the relevant agent tool with a clear 'message' describing the request.\n"
            "- Never attempt to call tools that are not listed verbatim above, even if they are mentioned in conversation by other agents.\n"
            "- If another agent lists its internal MCP tools, treat that as informational only—you must still message the agent via its agent.<name> tool to obtain results.\n"
            "- MCP tools can be called to gather information or perform actions as described.\n"
            "Rules:\n"
            "1. Respond using a single JSON object with one of the following shapes:\n"
            '   {"action": "final", "response": "<message for the user>"}\n'
            '   {"action": "tool", "tool_name": "<tool name>", "arguments": {...}}\n'
            "2. When calling a tool, only use the arguments defined in its schema.\n"
            "3. Use the full tool identifier (including server or agent prefix as shown above) when setting tool_name.\n"
            "4. For agent.<name> tools, provide a 'message' argument describing the request for that agent.\n"
            "5. After receiving tool results, provide a final response using the JSON format with action 'final'.\n"
            "6. Never include explanatory text outside of the JSON object.\n"
            "7. When the user asks about available tools or connections, list all items from the Available tools section.\n"
        )

    async def generate_reply(self, user_id: str, user_message: str) -> str:
        await self.ensure_ready()
        assert self._system_prompt is not None

        history = self._history.setdefault(user_id, [])
        messages = [{"role": "system", "content": self._system_prompt}, *history]
        messages.append({"role": "user", "content": user_message})
        logger.debug("Conversation history length for user %s: %d", user_id, len(history))
        logger.info("Prompting model %s with message: %s", self._model_name, user_message)

        assistant_response: Optional[str] = None
        iterations = 0
        raw_reply = ""

        while iterations < self._max_tool_iterations:
            raw_reply = await self._complete(messages)
            payload = _try_parse_json(raw_reply)

            if not payload:
                logger.debug("Iteration %d: no structured payload from model.", iterations + 1)
                assistant_response = raw_reply.strip()
                break

            action = _normalise_action(payload)
            if action == "final":
                logger.info("Iteration %d: model returned final response.", iterations + 1)
                assistant_response = payload.get("response", "").strip() or raw_reply.strip()
                break

            if action == "tool":
                tool_name = payload.get("tool_name")
                arguments = payload.get("arguments", {})

                if not isinstance(arguments, dict):
                    logger.warning("Iteration %d: tool %s received non-dict arguments %r", iterations + 1, tool_name, arguments)
                    assistant_response = "The tool arguments must be provided as a JSON object."
                    break

                # Always work with a mutable copy when enriching arguments.
                arguments = dict(arguments)
                payload["arguments"] = arguments

                if self._peer_client and self._peer_client.is_peer_tool(tool_name):
                    # Ensure a message string is present for peer-to-peer requests.
                    candidate_message = (
                        arguments.get("message")
                        or arguments.get("query")
                        or arguments.get("prompt")
                        or arguments.get("text")
                    )
                    if not isinstance(candidate_message, str) or not candidate_message.strip():
                        payload_message = payload.get("message")
                        if isinstance(payload_message, str) and payload_message.strip():
                            candidate_message = payload_message.strip()
                        else:
                            candidate_message = user_message.strip()
                        if not candidate_message:
                            candidate_message = (
                                "Please assist with the user's latest request; no additional details were provided."
                            )
                        arguments["message"] = candidate_message

                if not tool_name:
                    logger.warning("Iteration %d: tool call payload missing tool_name field", iterations + 1)
                    assistant_response = "Tool call missing 'tool_name'."
                    break

                try:
                    if self._peer_client and self._peer_client.is_peer_tool(tool_name):
                        tool_result = await self._peer_client.call_tool(tool_name, arguments)
                    else:
                        tool_result = await self._tool_client.call_tool(tool_name, arguments)
                except Exception as exc:  # pragma: no cover - surfaces operational issues
                    logging.exception("Tool execution failed: %s", exc)
                    assistant_response = (
                        f"Failed to execute tool '{tool_name}': {exc}"
                    )
                    break

                if not isinstance(tool_result, str):
                    tool_result = json.dumps(tool_result, ensure_ascii=False)

                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            f"Tool '{tool_name}' executed with arguments "
                            f"{json.dumps(arguments, ensure_ascii=False)} "
                            f"and returned:\n{tool_result}\n"
                            "Provide a final JSON response with action 'final'."
                        ),
                    }
                )
                iterations += 1
                continue

            logger.warning("Iteration %d: unknown action %s in payload %s", iterations + 1, action, payload)
            # Unknown action - treat as regular response
            assistant_response = raw_reply.strip()
            break

        if assistant_response is None:
            assistant_response = raw_reply.strip()

        # Persist conversation history for context-aware follow-ups
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": assistant_response})

        logger.info("Final response for user %s: %s", user_id, assistant_response)
        logger.debug("Cached conversation length for user %s is now %d", user_id, len(history))
        return assistant_response

    def run_in_loop(self, coro):
        """Execute coroutine in the agent's home event loop and return result."""
        if self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, self._loop)
            return future.result()
        return self._loop.run_until_complete(coro)

    async def _complete(self, messages: List[Dict[str, Any]]) -> str:
        if self._backend == "local":
            return await asyncio.to_thread(self._complete_local, messages)

        if not self._client:
            raise RuntimeError("Hugging Face client is not initialised.")

        try:
            response = await self._client.chat_completion(
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
            )
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Hugging Face model '{self._model_name}' is unavailable or requires additional access. "
                "Set HF_MODEL_ID to a reachable model or ensure your token has the required permissions."
            ) from exc
        choice = response.choices[0]
        message = choice.message
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)

        if isinstance(content, str):
            return content

        return str(content or "")

    def _complete_local(self, messages: List[Dict[str, Any]]) -> str:
        if not self._llama:
            raise RuntimeError("Local model backend is not initialised.")

        adapted_messages = [dict(entry) for entry in messages]
        if self._local_chat_format == "gemma":
            system_chunks: List[str] = []
            filtered: List[Dict[str, Any]] = []
            for msg in adapted_messages:
                role = msg.get("role")
                if role == "system":
                    system_chunks.append(str(msg.get("content", "")))
                    continue
                filtered.append(msg)

            if system_chunks:
                system_text = "\n\n".join(chunk for chunk in system_chunks if chunk)
                if filtered:
                    first = filtered[0]
                    merged_content = f"{system_text}\n\n{first.get('content', '')}".strip()
                    first["content"] = merged_content
                else:
                    filtered.append({"role": "user", "content": system_text})

                logger.debug(
                    "Gemma chat format does not support system messages; prepended instructions to first user turn."
                )

            adapted_messages = filtered

        try:
            response = self._llama.create_chat_completion(
                messages=adapted_messages,
                temperature=self._temperature,
                max_tokens=self._max_output_tokens,
            )
        except Exception as exc:
            logger.exception("Local model inference failed: %s", exc)
            raise

        choices = response.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(item.get("text", ""))
                else:
                    parts.append(str(item))
            return "".join(parts)

        if isinstance(content, str):
            return content

        return str(content or "")

class A2APeerClient:
    """Thin wrapper around python-a2a clients for tool-style usage."""

    TOOL_PREFIX = "agent."

    def __init__(self, peers: Iterable[Any]) -> None:
        self._peers = {
            getattr(peer, "name"): getattr(peer, "url")
            for peer in peers
            if getattr(peer, "name", None) and getattr(peer, "url", None)
        }
        self._clients: Dict[str, A2AClient] = {}

    def has_peers(self) -> bool:
        return bool(self._peers)

    def get_tool_descriptions(self) -> List[str]:
        if not self._peers:
            return []
        descriptions = []
        for name in sorted(self._peers):
            logger.debug("Peer tool registered: agent.%s -> %s", name, self._peers[name])
            descriptions.append(
                f"{self.TOOL_PREFIX}{name}: Relay a request to agent '{name}' via the A2A protocol "
                "(arguments: message: string (required), conversation_id: string (optional))."
            )
        return descriptions

    def is_peer_tool(self, tool_name: str) -> bool:
        if not isinstance(tool_name, str) or not self._peers:
            return False
        if not tool_name.startswith(self.TOOL_PREFIX):
            return False
        agent_name = tool_name[len(self.TOOL_PREFIX) :]
        return agent_name in self._peers

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self.is_peer_tool(tool_name):
            raise ValueError(f"Unknown peer tool '{tool_name}'.")

        agent_name = tool_name[len(self.TOOL_PREFIX) :]
        message = (
            arguments.get("message")
            or arguments.get("query")
            or arguments.get("prompt")
            or arguments.get("text")
        )
        if not message or not isinstance(message, str):
            raise ValueError("Peer tool requires a string 'message' argument.")

        client = self._clients.get(agent_name)
        if client is None:
            logger.info("Initialising A2A client for agent %s at %s", agent_name, self._peers[agent_name])
            peer_url = self._peers[agent_name].rstrip("/")
            client = A2AClient(peer_url, google_a2a_compatible=True)
            self._clients[agent_name] = client

        def _call_agent() -> str:
            logger.info("Calling peer agent %s", agent_name)
            logger.info("Peer agent %s request payload: %s", agent_name, message)
            response = client.ask(message)
            logger.info("Peer agent %s raw response: %s", agent_name, response)
            return response if isinstance(response, str) else str(response)

        try:
            return await asyncio.to_thread(_call_agent)
        except A2AError as exc:
            logger.exception("A2A protocol error when calling agent %s: %s", agent_name, exc)
            raise
        except Exception as exc:
            logger.exception("Unexpected error when calling agent %s: %s", agent_name, exc)
            raise

    def close(self) -> None:
        for name, client in list(self._clients.items()):
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    logger.debug("Closing A2A client for agent %s", name)
                    close_fn()
                except Exception:  # pragma: no cover - best effort cleanup
                    logging.exception("Failed to close A2A client", exc_info=True)
        self._clients.clear()


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON string that may be wrapped in Markdown fences."""
    stripped = text.strip()
    if not stripped:
        return None

    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        lines = stripped.splitlines()
        if lines and lines[0].lower() in {"json", "```json"}:
            lines = lines[1:]
        if lines and lines[-1] == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)

    try:
        candidate = json.loads(stripped)
        if isinstance(candidate, dict):
            return candidate
    except json.JSONDecodeError:
        try:
            start = stripped.index("{")
            end = stripped.rindex("}") + 1
            candidate = json.loads(stripped[start:end])
            if isinstance(candidate, dict):
                return candidate
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def _normalise_action(payload: Dict[str, Any]) -> str:
    """Map action/type fields from tool responses to canonical names."""
    action = (payload.get("action") or payload.get("type") or "").lower()
    if action in {"tool", "tool_call", "agent"}:
        return "tool"
    if action in {"final", "final_response", "respond"}:
        return "final"
    return action


@dataclass(slots=True)
class BotConfig:
    token: str
    allowed_users: Optional[Iterable[int]]
    parse_mode: ParseMode = ParseMode.MARKDOWN


class TelegramBot:
    def __init__(self, config: BotConfig, agent: LLMAgent) -> None:
        self._config = config
        self._agent = agent
        default_properties = (
            DefaultBotProperties(parse_mode=config.parse_mode)
            if DefaultBotProperties and config.parse_mode
            else None
        )

        if default_properties:
            self._bot = Bot(token=config.token, default=default_properties)
        else:
            self._bot = Bot(token=config.token, parse_mode=config.parse_mode)
        self._dispatcher = Dispatcher()
        self._allowed_users = (
            {int(uid) for uid in config.allowed_users} if config.allowed_users else None
        )

        self._dispatcher.message.register(self._on_start, CommandStart())
        self._dispatcher.message.register(self._on_message)

    async def run(self) -> None:
        logger.info("Telegram bot initialisation started.")
        await self._agent.ensure_ready()
        logger.info("LLM agent and MCP clients ready.")
        try:
            me = await self._bot.get_me()
            username = getattr(me, "username", None) or getattr(me, "first_name", "unknown")
            logger.info("Telegram bot polling starting as @%s (id=%s)", username, getattr(me, "id", "n/a"))
        except Exception:  # pragma: no cover - logging aid; ignore failures
            logger.info("Telegram bot polling starting (failed to fetch bot profile).")
        try:
            logger.info("Dispatcher polling loop starting.")
            await self._dispatcher.start_polling(self._bot)
        finally:
            logger.info("Telegram bot polling stopped.")
            await self._bot.session.close()

    async def _on_start(self, message: Message) -> None:
        logger.info("/start received in chat %s from user %s", message.chat.id, getattr(message.from_user, "id", None))
        if not await self._is_allowed(message):
            return
        await message.answer("Привет! Отправь мне вопрос — я подключу инструменты и помогу.")

    async def _on_message(self, message: Message) -> None:
        logger.info("Telegram message received: chat=%s user=%s text=%r", message.chat.id, getattr(message.from_user, "id", None), message.text)
        if not await self._is_allowed(message):
            return

        chat_id = message.chat.id
        if not message.text:
            logger.info("Sending empty-message warning to chat %s", chat_id)
            await message.answer("Я могу работать только с текстовыми сообщениями.")
            return

        await self._bot.send_chat_action(message.chat.id, ChatAction.TYPING)
        try:
            reply = await self._agent.generate_reply(str(message.from_user.id), message.text)
        except Exception as exc:  # pragma: no cover - surfaces operational issues
            logging.exception("Failed to process message in chat %s: %s", chat_id, exc)
            await message.answer("Произошла ошибка при обработке запроса. Попробуй снова позже.")
            return

        try:
            logger.info("Sending reply to chat %s", chat_id)
            await message.answer(reply)
        except TelegramBadRequest:
            logger.warning("Telegram could not parse reply for chat %s, resending without parse mode.", chat_id)
            await message.answer(reply, parse_mode=None)

    async def _is_allowed(self, message: Message) -> bool:
        if self._allowed_users is None:
            logger.debug("No allow list configured; permitting message in chat %s.", message.chat.id)
            return True

        user_id = message.from_user.id if message.from_user else None
        if user_id is None:
            logger.warning("Could not determine user id for chat %s", message.chat.id)
            await message.answer("Не удалось определить пользователя.")
            return False

        if user_id not in self._allowed_users:
            logger.warning("Blocked message from unauthorized user: %s", user_id)
            await message.answer("У тебя нет доступа к этому боту.")
            return False

        logger.debug("User %s is allowed to interact with the bot.", user_id)
        return True
