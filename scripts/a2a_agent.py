"""
Bridge layer between the local Gemma-powered assistant and the Agent-to-Agent (A2A) protocol.

This module exposes a thin wrapper around ``python-a2a`` so that our existing ``LLMAgent``
can be presented as an A2A server. Tasks that arrive over the protocol are converted into
plain text prompts, delegated to ``LLMAgent.generate_reply`` (ensuring MCP tools remain
available), and then translated back into A2A artifacts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from python_a2a import (
    A2AServer,
    AgentCard,
    AgentSkill,
    Message,
    MessageRole,
    Task,
    TaskState,
    TaskStatus,
    TextContent,
    run_server,
)
from python_a2a.server.http import create_flask_app

import asyncio

from telegram_bot import LLMAgent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class A2AAgentConfig:
    """Configuration block for the A2A wrapper."""

    name: str
    description: str
    version: str
    url: str
    skill_name: str = "Chat"
    skill_description: str = "Conversational interface backed by Gemma via MCP tools."


class TelegramBridgeA2AServer(A2AServer):
    """
    A2A server that delegates task resolution to the existing LLMAgent.

    Each conversation is keyed by the requester-provided conversation ID (when available);
    otherwise a synthetic identifier is generated from the task.
    """

    def __init__(self, llm_agent: LLMAgent, config: A2AAgentConfig) -> None:
        agent_card = AgentCard(
            name=config.name,
            description=config.description,
            url=config.url,
            version=config.version,
            skills=[
                AgentSkill(
                    name=config.skill_name,
                    description=config.skill_description,
                    tags=["chat", "conversation", "mcp"],
                    examples=[
                        "Help me plan my day using available tools.",
                        "What time is it in Tokyo right now?",
                        "Generate a random number between 1 and 10.",
                    ],
                )
            ],
            capabilities={
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": False,
                "google_a2a_compatible": True,
                "parts_array_format": True,
            },
        )

        super().__init__(agent_card=agent_card)
        self._llm_agent = llm_agent

    def handle_task(self, task: Task) -> Task:
        """Synchronously handle incoming tasks by delegating to the LLMAgent."""
        logger.info("Received A2A task %s", task.id)

        user_message = self._extract_text_from_task(task)
        if not user_message:
            logger.warning("Task %s contained no textual content; returning default reply.", task.id)
            task.artifacts = [
                {"parts": [{"type": "text", "text": "I can only process textual requests at the moment."}]}
            ]
            task.status = TaskStatus(state=TaskState.COMPLETED)
            return task

        conversation_id = self._determine_conversation_id(task)
        logger.info(
            "Delegating task %s (conversation %s) with message: %s",
            task.id,
            conversation_id,
            user_message,
        )

        try:
            response_text = self._llm_agent.run_in_loop(
                self._llm_agent.generate_reply(conversation_id, user_message)
            )
        except Exception as exc:  # pragma: no cover - surfaces runtime issues
            logger.exception("Failed to generate reply for conversation %s: %s", conversation_id, exc)
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"error": str(exc)},
            )
            return task

        task.artifacts = [{"parts": [{"type": "text", "text": response_text}]}]
        task.status = TaskStatus(state=TaskState.COMPLETED)
        logger.info("Task %s completed successfully.", task.id)
        return task

    @staticmethod
    def _extract_text_from_task(task: Task) -> str:
        message_payload = task.message or {}
        if isinstance(message_payload, Message):
            content = message_payload.content
            if isinstance(content, TextContent):
                return content.text or ""
            if hasattr(content, "text"):
                return getattr(content, "text") or ""
            return ""

        if isinstance(message_payload, dict):
            content = message_payload.get("content")
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str):
                    return text
            elif isinstance(content, TextContent):
                return content.text or ""
        return ""

    @staticmethod
    def _determine_conversation_id(task: Task) -> str:
        """
        Derive a stable conversation identifier from the task payload.

        The python-a2a Task model does not expose a conversation_id field, so we
        have to look through the task and message metadata defensively. This
        keeps the same value when provided by the caller and falls back to a
        synthetic identifier when none is available.
        """
        task_conv_id = getattr(task, "conversation_id", None) or getattr(task, "conversationId", None)
        if task_conv_id:
            return str(task_conv_id)

        message_payload = getattr(task, "message", None)
        if isinstance(message_payload, Message):
            message_conv_id = getattr(message_payload, "conversation_id", None) or getattr(
                message_payload, "conversationId", None
            )
            if message_conv_id:
                return str(message_conv_id)
        elif isinstance(message_payload, dict):
            message_conv_id = message_payload.get("conversationId") or message_payload.get("conversation_id")
            if message_conv_id:
                return str(message_conv_id)

            content = message_payload.get("content")
            if isinstance(content, dict):
                content_conv_id = content.get("conversationId") or content.get("conversation_id")
                if content_conv_id:
                    return str(content_conv_id)

        return f"a2a-{task.id}"


def run_a2a_server(llm_agent: LLMAgent, config: A2AAgentConfig, host: str, port: int) -> None:
    """
    Convenience helper to start the A2A server using the provided configuration.
    """
    server = TelegramBridgeA2AServer(llm_agent=llm_agent, config=config)
    logger.info(
        "Starting A2A server '%s' on %s:%s (%s)",
        config.name,
        host,
        port,
        config.url,
    )
    try:
        app = create_flask_app(server)
    except Exception:  # pragma: no cover - fallback to default server behaviour
        logger.warning("Falling back to default python-a2a server; create_flask_app failed.", exc_info=True)
        run_server(server, host=host, port=port)
        return

    print(f"Starting A2A server on http://{host}:{port}/a2a")
    google_capabilities = getattr(getattr(server, "agent_card", None), "capabilities", {}) or {}
    google_enabled = google_capabilities.get("google_a2a_compatible") or google_capabilities.get("parts_array_format")
    print(f"Google A2A compatibility: {'Enabled' if google_enabled else 'Disabled'}")

    # Ensure Flask runs in the same thread so asyncio objects stay on their owning loop.
    app.run(host=host, port=port, debug=False, threaded=False)
