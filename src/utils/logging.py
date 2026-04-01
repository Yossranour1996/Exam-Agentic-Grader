# src/utils/logging.py
import logging
from typing import Sequence
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.utils import io


def get_logger(path: str | Path) -> logging.Logger:
	"""
    Sets up and returns a Python logger dedicated to a specific student's run path.
    """
	log_path = Path(path)
	logger = logging.getLogger(log_path.parent.name)

	if not logger.handlers:
		log_path.parent.mkdir(parents=True, exist_ok=True)

		# Create a file handler
		file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')

		formatter = logging.Formatter(
			'%(asctime)s | %(levelname)-7s\n%(message)s', 
			datefmt='%Y-%m-%d %H:%M:%S'
		)

		file_handler.setLevel(logging.DEBUG)
		file_handler.setFormatter(formatter)
		
		logger.setLevel(logging.DEBUG)
		logger.addHandler(file_handler)

		return logger
	
	return logger

def get_log_entry(messages: Sequence[BaseMessage], step_label: str) -> str:
	"""
	Combines a sequence of messages into a single log entry.
	"""
	log_entry = f"{'=' * 80}\n[{step_label}] Message log:\n\n"
	for message in messages:
		if isinstance(message, SystemMessage):
			content = io.truncate_text(str(message.content), limit=40)
			log_entry += (f"{'-' * 60}\nSYSTEM: {content}\n\n")

		elif isinstance(message, HumanMessage):
			content = io.truncate_text(str(message.content), limit=40)
			log_entry += (f"{'-' * 60}\nHUMAN : {content}\n\n")

		elif isinstance(message, AIMessage):
			# If the AI used a tool, log it distinctly
			if hasattr(message, "tool_calls") and message.tool_calls:
				log_entry += (f"{'-' * 60}\nAI TOOL CALL: {message.tool_calls}\n\n")
			else:
				log_entry += (f"{'-' * 60}\nAI    : {message.content}\n\n")

		elif isinstance(message, ToolMessage):
			log_entry += (f"{'-' * 60}\nTOOL  : {message.content}\n\n")

		else:
			log_entry += (f"{'-' * 60}\nOTHER : {message.content}\n\n")
		
	return log_entry


def log_messages(
		messages: Sequence[BaseMessage],
		path: str | Path,
		step_label: str = "step"
		) -> None:
	"""
	Log messages in the specified directory.

	:param messages: Sequence of BaseMessage objects to log
	:param path: file path where the message log will be saved
	:param step_label: Grading step to log messages under.
	"""
	logger = get_logger(path)
	entry = get_log_entry(messages, step_label)
	logger.info(entry)
