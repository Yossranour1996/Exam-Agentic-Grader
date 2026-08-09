# other_architectures/single_agent/utils/io.py
"""Small, consistent file I/O helpers."""

import json
from base64 import b64encode
from pathlib import Path
from typing import Any

import yaml
from openpyxl import Workbook, load_workbook

from single_agent.utils.logging import Logger


def _error(message: str, error: Exception, logger: Logger | None) -> None:
    text = f"{message}: {error}"
    print(text)
    if logger:
        logger.log(text, level="error")


def read_text(filepath: str | Path, logger: Logger | None = None, encoding="utf-8") -> str | None:
    """"
    Read the contents of a text file and return it as a string.
    
    Args:
        filepath: Path to the text file.
        logger: Optional Logger instance for logging errors.
        encoding: Encoding to use when reading the file (default is 'utf-8').

    Returns:
            The contents of the file as a string, or None if an error occurred.
    """
    try:
        return Path(filepath).read_text(encoding=encoding)
    except Exception as error:
        _error(f"Error reading {filepath}", error, logger)
        return None


def write_text(filepath: str | Path, content: str, logger: Logger | None = None, encoding="utf-8", append=False) -> bool:
    """"
    Write content to a text file.

    Args:
        filepath: Path to the text file.
        content: The string to write to the file.
        logger: Optional Logger instance for logging errors.
        encoding: Encoding to use when writing the file (default is 'utf-8').
        append: If True, append to the file instead of overwriting it.

    Returns:
        True if the operation was successful, False otherwise.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if append else "w", encoding=encoding) as file:
            file.write(content)
        return True
    except Exception as error:
        _error(f"Error writing {filepath}", error, logger)
        return False


def read_json(filepath: str | Path, logger: Logger | None = None) -> Any | None:
    """"
    Read a JSON file and return its contents as a Python object.

    Args:
        filepath: Path to the JSON file.
        logger: Optional Logger instance for logging errors.

    Returns:
        The parsed JSON data as a Python object, or None if an error occurred.
    """
    try:
        text = Path(filepath).read_text(encoding="utf-8").strip()
        return json.loads(text) if text else None
    except FileNotFoundError:
        return None
    except Exception as error:
        _error(f"Error parsing JSON from {filepath}", error, logger)
        return None


def read_yaml(filepath: str | Path, logger: Logger | None = None) -> Any | None:
    """"
    Read a YAML file and return its contents as a Python object.

    Args:
        filepath: Path to the YAML file.
        logger: Optional Logger instance for logging errors.

    Returns:
        The parsed YAML data as a Python object, or None if an error occurred.
    """
    try:
        return yaml.safe_load(Path(filepath).read_text(encoding="utf-8"))
    except Exception as error:
        _error(f"Error parsing YAML from {filepath}", error, logger)
        return None


def write_json(filepath: str | Path, data: Any, logger: Logger | None = None, indent=2, ensure_ascii=False) -> bool:
    """"
    Write a Python object to a JSON file.

    Args:
        filepath: Path to the JSON file.
        data: The Python object to write to the file.
        logger: Optional Logger instance for logging errors.
        indent: The number of spaces to use for indentation (default is 2).
        ensure_ascii: If True, escape non-ASCII characters (default is False).

    Returns:
        True if the operation was successful, False otherwise.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        final = data
        existing = read_json(path, logger) if path.exists() else None
        if isinstance(existing, list):
            final = existing + (data if isinstance(data, list) else [data])
        path.write_text(json.dumps(final, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")
        return True
    except Exception as error:
        _error(f"Error writing JSON to {filepath}", error, logger)
        return False


def read_excel(filepath: str | Path, logger: Logger | None = None) -> dict[str, list[dict[str, Any]]] | None:
    """"
    Read an Excel file and return its contents as a dictionary of sheets.

    Args:
        filepath: Path to the Excel file.
        logger: Optional Logger instance for logging errors.

    Returns:
        A dictionary where keys are sheet names and values are lists of dictionaries representing rows, or None if an error occurred.
    """
    try:
        workbook = load_workbook(filepath, data_only=True)
        result = {}
        for name in workbook.sheetnames:
            rows = list(workbook[name].iter_rows(values_only=True))
            headers = [str(value) if value is not None else f"Column_{i}" for i, value in enumerate(rows[0])] if rows else []
            result[name] = [dict(zip(headers, row)) for row in rows[1:]]
        return result
    except Exception as error:
        _error(f"Error reading Excel file {filepath}", error, logger)
        return None


def write_excel(filepath: str | Path, data: dict[str, list[dict[str, Any]]], logger: Logger | None = None) -> bool:
    """"
    Write data to an Excel file.

    Args:
        filepath: Path to the Excel file.
        data: A dictionary where keys are sheet names and values are lists of dictionaries representing rows.
        logger: Optional Logger instance for logging errors.

    Returns:
        True if the operation was successful, False otherwise.
    """
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_excel(path, logger) if path.exists() else {}
        combined = {name: (existing or {}).get(name, []) + rows for name, rows in data.items()}
        for name, rows in (existing or {}).items():
            combined.setdefault(name, rows)
        workbook = Workbook()
        workbook.remove(workbook.active)
        for name, rows in combined.items():
            sheet = workbook.create_sheet(name)
            if rows:
                headers = list(dict.fromkeys(key for row in rows for key in row))
                sheet.append(headers)
                for row in rows:
                    sheet.append([row.get(header, "") for header in headers])
        workbook.save(path)
        return True
    except Exception as error:
        _error(f"Error writing Excel file {filepath}", error, logger)
        return False


def read_base64(filepath: str | Path, logger: Logger | None = None) -> str | None:
    """"
    Read a file and return its base64-encoded string.

    Args:
        filepath: Path to the file.
        logger: Optional Logger instance for logging errors.

    Returns:
        The base64-encoded string, or None if an error occurred.
    """
    try:
        return b64encode(Path(filepath).read_bytes()).decode("ascii")
    except Exception as error:
        _error(f"Error reading {filepath} as base64", error, logger)
        return None


def file_exists(filepath: str | Path) -> bool:
    return Path(filepath).exists()


def delete_file(filepath: str | Path, logger: Logger | None = None) -> bool:
    try:
        Path(filepath).unlink(missing_ok=True)
        return True
    except Exception as error:
        _error(f"Error deleting {filepath}", error, logger)
        return False
