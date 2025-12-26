from __future__ import annotations

import os
import logging
import traceback
from datetime import datetime as _dt
from typing import Optional

_LOG_FILE_PATH: Optional[str] = None

def _ensure_logs_dir() -> str:
    """Ensure the 'logs' directory exists and return its path."""
    base = os.getcwd()
    logs_dir = os.path.join(base, "logs")
    try:
        os.makedirs(logs_dir, exist_ok=True)
    except Exception:
        pass
    return logs_dir

def _open_log_file() -> str:
    """Get the current day's log file path, creating it if necessary."""
    global _LOG_FILE_PATH

    logs_dir = _ensure_logs_dir()
    # Use daily log files as requested
    date_str = _dt.now().strftime("%Y%m%d")
    expected_path = os.path.join(logs_dir, f"{date_str}.txt")

    # Refresh path if day changed
    if _LOG_FILE_PATH != expected_path:
        _LOG_FILE_PATH = expected_path

    return _LOG_FILE_PATH

def _log(line: str) -> None:
    """Write to logfile only for warnings/errors."""
    try:
        u = str(line).upper()
        if ("ERROR" not in u) and ("WARN" not in u):
            return
        path = _open_log_file()
        with open(path, "a", encoding="utf-8") as f:
            ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {line.rstrip('\n')}\n")
    except Exception:
        pass

def _log_exception(msg: str, exc: Exception) -> None:
    """Log an exception with its full stacktrace."""
    try:
        path = _open_log_file()
        with open(path, "a", encoding="utf-8") as f:
            ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] ERROR: {msg}\n")
            f.write(traceback.format_exc())
            f.write("\n" + "-"*40 + "\n")
    except Exception:
        pass

def _suppress_ff1_logs() -> None:
    """Ensure FastF1 loggers do not output to the console and redirect to file."""
    try:
        import fastf1

        logs_dir = _ensure_logs_dir()
        log_file = _open_log_file()

        ff1_logger = logging.getLogger("fastf1")
        ff1_logger.setLevel(logging.WARNING)
        fastf1.set_log_level("WARNING")

        has_file_handler = any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file) for h in ff1_logger.handlers)

        if not has_file_handler:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            ff1_logger.addHandler(fh)

        for h in ff1_logger.handlers[:]:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                h.setLevel(100)

        for name in logging.root.manager.loggerDict:
            if name.startswith("fastf1"):
                logging.getLogger(name).setLevel(logging.WARNING)

    except Exception:
        pass
