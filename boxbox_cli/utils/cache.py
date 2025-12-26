import os
from .logger import _suppress_ff1_logs

try:
    from fastf1 import Cache as _FF1Cache  # type: ignore

    def _enable_ff1_cache() -> None:
        """Initialize and enable the FastF1 cache in the project root or a custom location."""
        _suppress_ff1_logs()

        cache_dir = os.getenv("FASTF1_CACHE_DIR")
        if not cache_dir:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(project_root, "cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        try:
            # Ensure we use an absolute path for the cache directory
            abs_cache_dir = os.path.abspath(cache_dir)

            # Integrity check for requests-cache sqlite file
            sqlite_path = os.path.join(abs_cache_dir, "fastf1_http_cache.sqlite")
            if os.path.exists(sqlite_path):
                # If file is very small (e.g. 0 bytes), it might be corrupted.
                # Also, we might want to check for other corruption signs if possible.
                if os.path.getsize(sqlite_path) == 0:
                    try:
                        os.remove(sqlite_path)
                    except Exception:
                        pass
                else:
                    # Simple check to see if it's a valid sqlite file
                    try:
                        import sqlite3
                        conn = sqlite3.connect(sqlite_path)
                        conn.execute("SELECT 1 FROM responses LIMIT 1")
                        conn.close()
                    except Exception:
                        # If it's corrupted, remove it
                        try:
                            conn.close()
                        except Exception:
                            pass
                        try:
                            os.remove(sqlite_path)
                        except Exception:
                            pass

            if _FF1Cache._CACHE_DIR == abs_cache_dir:
                return
            _FF1Cache.enable_cache(abs_cache_dir)
        except Exception:
            pass

except Exception:
    pass
