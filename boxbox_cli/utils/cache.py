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
            if _FF1Cache._CACHE_DIR == str(cache_dir):
                return
            _FF1Cache.enable_cache(cache_dir)
        except Exception:
            pass

except Exception:
    pass
