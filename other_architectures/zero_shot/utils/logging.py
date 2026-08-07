# other_architectures/zero_shot/utils/logging.py
"""Per-run logging with readable LLM message traces."""

import logging
from pathlib import Path
from typing import Literal, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


class Logger:
    levels = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "debug": logging.DEBUG}

    def __init__(self, path: str | Path):
        """
        A simple logger that writes log entries to a specified file with timestamps and severity levels.
        It also provides methods to log sequences of messages and retrieve formatted message logs.
        """
        log_path = Path(path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(str(log_path.resolve()))
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()
        self.logger.propagate = False
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s\n%(message)s", "%Y-%m-%d %H:%M:%S"))
        self.logger.addHandler(handler)

    def log(self, entry: str, level: Literal["info", "warning", "error", "debug"] = "info", step_label="log") -> None:
        """
        Log an entry to the log file with a specified level and step label.
        
        Args:
            entry: The log message to be recorded.
            level: The severity level of the log message (default is "info").
            step_label: The label for the step being logged.
        """
        self.logger.log(self.levels.get(level, logging.INFO), f"{'=' * 80}\n[{step_label}]: {entry}")

    def get_messages_entry(self, messages: Sequence[BaseMessage], step_label: str) -> str:
        output = f"{'=' * 80}\n[{step_label}] Message log:\n\n"
        labels = {SystemMessage: "SYSTEM", HumanMessage: "HUMAN", AIMessage: "AI", ToolMessage: "TOOL"}
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                content = "\n".join(f"AI TOOL CALL: {call}" for call in message.tool_calls)
            else:
                label = next((value for kind, value in labels.items() if isinstance(message, kind)), "OTHER")
                content = f"{label}: {truncate_text(str(message.content))}"
            output += f"{'-' * 60}\n{content}\n\n"
        return output

    def log_messages(self, messages: Sequence[BaseMessage], step_label="step") -> None:
        """
        Log a sequence of messages to the log file with a specified step label.
        
        Args:
            messages: A sequence of BaseMessage instances to be logged.
            step_label: The label for the step being logged.
        """
        self.logger.info(self.get_messages_entry(messages, step_label))

    def shutdown(self) -> None:
        """
        Shutdown the logger and close all handlers.
        """
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def truncate_text(text: str, max_lines=40) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    half = max_lines // 2
    return "\n".join(lines[:half] + [".", ".", "."] + lines[-half:])
