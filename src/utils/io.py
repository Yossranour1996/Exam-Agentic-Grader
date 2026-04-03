# src/utils/io.py
"""Unified file I/O utilities for reading and writing various file types."""

import json
from base64 import b64encode
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


def read_json(filepath: str | Path) -> Any | None:
    """
    Safely read and parse JSON file.
    
    :param filepath: Path to JSON file
    :return: Parsed JSON object or None if error
    """
    path = Path(filepath)
    try:
        content = path.read_text(encoding="utf-8")
        js = json.loads(content) if len(content) > 0 else []
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
    if path.exists():
    # If file exists, read existing content and
    # append new data to it before writing back.
        try:
            content = path.read_text(encoding="utf-8")
            js = json.loads(content) if len(content) > 0 else []
            data = (js if isinstance(js, list) else [js]) + data
        except Exception as e:
            print(f"Error writing JSON to {filepath}: {e}")
            return False
    try:
        with open(path, "w", encoding="utf-8")  as file:
            file.write(
                json.dumps(data, ensure_ascii=ensure_ascii, indent=indent),
                )
        return True
    except Exception as e:
        print(f"Error writing JSON to {filepath}: {e}")
        return False


def read_base64(filepath: str | Path) -> str | None:
    """Safely read file and return its content as base64 string.
    
    :param filepath: Path to file
    :return: Base64 string of file content or None if error
    """
    path = Path(filepath)
    try:
        file_b64 = path.read_bytes()
        return b64encode(file_b64).decode("utf-8")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def truncate_text(text: str, limit: int = 40) -> str:
    """
    Truncate text by keeping the first and last parts, 
    and replacing the middle with ellipsis if it exceeds the line limit.

    :param text: Text to truncate.
    :param limit: Min text lenght to apply truncation.
    :return: Truncated text if truncation applied.
    """
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


def delete_file(filepath: str | Path) -> bool:
    """
    Safely delete file if it exists.
    
    :param filepath: Path to file
    :return: True if deleted or not exist, False if error
    """
    path = Path(filepath)
    try:
        if path.exists():
            path.unlink()
        return True
    except Exception as e:
        print(f"Error deleting {filepath}: {e}")
        return False
