# src/utils/io.py
"""Unified file I/O utilities for reading and writing various file types."""

import json
from pathlib import Path
from typing import Any, List, Dict


def read_text(filepath: str | Path, encoding: str = "utf-8") -> str | None:
    """
    Safely read text from file.
    
    :param filepath: Path to text file
    :param encoding: File encoding (default: utf-8)
    :return: File contents or None if error
    """
    path = Path(filepath)
    try:
        return path.read_text(encoding=encoding)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def write_text(filepath: str | Path, content: str, encoding: str = "utf-8") -> bool:
    """
    Safely write text to file.
    
    :param filepath: Path to text file
    :param content: Content to write
    :param encoding: File encoding (default: utf-8)
    :return: True if successful, False otherwise
    """
    path = Path(filepath)
    try:
        with open(path, "a", encoding=encoding)  as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error writing to {filepath}: {e}")
        return False


def read_json(filepath: str | Path) -> List[Dict[str, Any]] | None:
    """
    Safely read and parse JSON file.
    
    :param filepath: Path to JSON file
    :return: Parsed JSON list of dicts or None if error
    """
    path = Path(filepath)
    try:
        content = path.read_text(encoding="utf-8")
        js = json.loads(content)
        return js
    except Exception as e:
        print(f"Error parsing JSON from {filepath}: {e}")
        return None


def write_json(filepath: str | Path, data: List[Dict[str, Any]], indent: int = 2, ensure_ascii: bool = False) -> bool:
    """
    Safely write dict as JSON file.
    
    :param filepath: Path to JSON file
    :param data: List of dictionaries to write
    :param indent: JSON indentation level (default: 2)
    :param ensure_ascii: Whether to escape non-ASCII characters (default: False)
    :return: True if successful, False otherwise
    """
    path = Path(filepath)
    try:
        with open(path, "a", encoding="utf-8")  as file:
            file.write(
                json.dumps(data, ensure_ascii=ensure_ascii, indent=indent),
            )
        return True
    except Exception as e:
        print(f"Error writing JSON to {filepath}: {e}")
        return False


def truncate_text(text: str, limit: int = 40) -> str:
	list_of_lines = text.split("\n")
	length = len(list_of_lines)

	if length > limit:
		return (
			'\n'.join(list_of_lines[:(limit // 2)]) + 
			"\n.\n.\n.\n" + 
			'\n'.join(list_of_lines[(length - (limit // 2)):])
			)
	return text


def file_exists(filepath: str | Path) -> bool:
    """
    Check if file exists.
    
    :param filepath: Path to check
    :return: True if file exists, False otherwise
    """
    return Path(filepath).exists()
