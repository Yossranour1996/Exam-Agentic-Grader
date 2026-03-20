# src/utils/logging.py
from typing import Sequence
from pathlib import Path
from datetime import datetime

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from src.utils import io

def log(messages: Sequence[BaseMessage], path: str | Path, step_label: str = "step") -> None:
	"""
	Log messages in the specified directory.

	:param messages: Sequence of BaseMessage objects to log
	:param path: file path where the message log will be saved
	"""
	log_path = Path(path)
	# Start the log if first log entry
	if not log_path.exists():
		log_path.parent.mkdir(parents=True, exist_ok=True)
		try:
			with open(log_path, "w", encoding="utf-8") as file:
				file.write("Message Log:\n\n")
		except Exception as e:
			print(f"Error creating log file: {e}")
			return

	try:
		with open(log_path, "a", encoding="utf-8") as file:
			# Start of a log entry
			file.write(f"\n[{datetime.now().strftime("%B %d, %H:%M:%S")}] [{step_label}] Message log entry:\n")
			file.write("=" * 100 + "\n")

			# Log by message type
			for message in messages:
				if isinstance(message, SystemMessage):
					file.write(f"\n\tSystem:\n\n{io.truncate_text(message.content)}\n\n" + "-" * 60 + "\n")
				elif isinstance(message, HumanMessage):
					file.write(f"\n\tUser:\n\n{io.truncate_text(message.content)}\n\n" + "-" * 60 + "\n")
				elif isinstance(message, AIMessage):
					file.write(f"\n\tAI:\n\n{io.truncate_text(message.content)}\n\n" + "-" * 60 + "\n")
			file.write("\n")
	except Exception as e:
		print(f"Error writing log: {e}")
