
import os
import shutil
import threading
import time
from boxbox_cli.utils.stats import _aggregate_season_stats, prewarm_driver_enrichment, complete_season_summaries, shutdown_background_tasks

def test_race_condition():
    # Clear cache/stats
    project_root = os.getcwd()
    stats_cache_dir = os.path.join(project_root, "cache", "stats")
    if os.path.exists(stats_cache_dir):
        shutil.rmtree(stats_cache_dir)
    
    year = 2024
    # Mock driver abbreviations for 2024
    abbrs = ["VER", "NOR", "LEC", "PIA", "SAI", "HAM", "RUS", "PER", "ALO", "HUL"]

    print(f"Starting concurrent aggregation and enrichment for {year}...")
    
    def run_aggregation():
        print("Thread: Starting _aggregate_season_stats")
        _aggregate_season_stats(year)
        print("Thread: Finished _aggregate_season_stats")

    def run_prewarm():
        print("Thread: Starting prewarm_driver_enrichment")
        prewarm_driver_enrichment(year, abbrs)
        print("Thread: Finished prewarm_driver_enrichment")

    def run_complete():
        print("Thread: Starting complete_season_summaries")
        complete_season_summaries(year)
        print("Thread: Finished complete_season_summaries")

    threads = [
        threading.Thread(target=run_aggregation),
        threading.Thread(target=run_prewarm),
        threading.Thread(target=run_complete)
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print("All threads finished. Checking stats...")
    drivers, teams = _aggregate_season_stats(year)
    
    # Check if Verstappen has points (he should if aggregation worked)
    ver_pts = drivers.get("VER", {}).get("points", 0)
    print(f"VER Points: {ver_pts}")
    
    if ver_pts > 0:
        print("SUCCESS: Stats seem populated and consistent.")
    else:
        print("FAILURE: Stats are 0. Data might be corrupted or aggregation failed.")

    shutdown_background_tasks()

if __name__ == "__main__":
    test_race_condition()
