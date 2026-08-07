from __future__ import annotations
import os
import time
import threading
from typing import Optional, Dict, Any, Callable, Self
import psutil

#> ---------------------------------------------------------------------------------------

class MemoryTracker:
    """
    Real-time process-tree memory tracker.
    
    Monitors RSS (Resident Set Size) memory across the main Python process 
    and all child processes (e.g., joblib/loky worker processes) by sampling 
    in a background thread.
    """
    def __init__(self, sample_interval_sec: float = 0.05):
        """
        :param sample_interval_sec: How frequently (in seconds) to sample RAM. 
                                    0.05s (50ms) gives high precision with negligible overhead.
        """
        self.sample_interval = sample_interval_sec
        self.pid = os.getpid()
        self.parent_proc = psutil.Process(self.pid)
        
        self._stop_signal = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # --- Metrics (Stored in Bytes) ---
        self.baseline_rss: int = 0
        self.peak_rss: int = 0
        self.final_rss: int = 0
        self.sum_rss: int = 0      # ADDED: To track total for average
        self.samples_taken: int = 0

    def _get_process_tree_rss(self) -> int:
        """
        Calculates total RSS (Resident Set Size) in bytes across parent and all active children.
        """
        total_bytes = 0
        try:
            # Main parent process memory
            total_bytes += self.parent_proc.memory_info().rss
            
            # Fetch all child processes spawned by joblib/loky
            children = self.parent_proc.children(recursive=True)
            for child in children:
                try:
                    if child.is_running():
                        total_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Child process finished or terminated during loop
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        return total_bytes

    def _sample_loop(self) -> None:
        """Background thread target that updates RAM metrics periodically."""
        # FIXED: Use wait() instead of time.sleep() for immediate shutdown
        while not self._stop_signal.wait(self.sample_interval):
            current_rss = self._get_process_tree_rss()
            if current_rss > self.peak_rss:
                self.peak_rss = current_rss
            self.sum_rss += current_rss  # ADDED: Accumulate for average
            self.samples_taken += 1


    def start(self) -> Self:
        """Starts background memory profiling."""
        self.baseline_rss = self._get_process_tree_rss()
        self.peak_rss = self.baseline_rss
        self._stop_signal.clear()
        
        self._monitor_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._monitor_thread.start()
        return self


    def stop(self) -> Self:
        """Stops profiling and captures final RAM state."""
        self._stop_signal.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        
        self.final_rss = self._get_process_tree_rss()
        # Final sanity check in case peak happened at the exact moment of stopping
        if self.final_rss > self.peak_rss:
            self.peak_rss = self.final_rss
        return self


    # --- Context Manager Support ---
    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # --- Metric Output Properties ---
    @property
    def peak_mb(self) -> float:
        """Peak memory used in Megabytes."""
        return self.peak_rss / (1024 * 1024)

    @property
    def peak_gb(self) -> float:
        """Peak memory used in Gigabytes."""
        return self.peak_rss / (1024 * 1024 * 1024)

    @property
    def net_mb(self) -> float:
        """Net increase in RAM (Peak RAM minus Baseline RAM) in MB."""
        return max(0.0, (self.peak_rss - self.baseline_rss) / (1024 * 1024))

    @property
    def net_gb(self) -> float:
        """Net increase in RAM (Peak RAM minus Baseline RAM) in GB."""
        return max(0.0, (self.peak_rss - self.baseline_rss) / (1024 * 1024 * 1024))

    @property
    def avg_mb(self) -> float:
        """Average memory used in Megabytes."""
        if self.samples_taken == 0:
            return 0.0
        return (self.sum_rss / self.samples_taken) / (1024 * 1024)

    @property
    def net_avg_mb(self) -> float:
        """Net average increase in RAM (Average RAM minus Baseline RAM) in MB."""
        baseline_mb = self.baseline_rss / (1024 * 1024)
        return max(0.0, self.avg_mb - baseline_mb)


    def summary(self) -> Dict[str, Any]:
        """Returns a clean report dictionary."""
        return {
            "baseline_mb": round(self.baseline_rss / (1024 * 1024), 2),
            "peak_mb": round(self.peak_mb, 2),
            "avg_mb": round(self.avg_mb, 2),
            "peak_gb": round(self.peak_gb, 3),
            "net_peak_mb": round(self.net_mb, 2),
            "net_avg_mb": round(self.net_avg_mb, 2),
            "samples_taken": self.samples_taken
        }


    def print_report(self, label: str = "Execution") -> None:
        """Prints a human-readable profile report."""
        print(f"\n--- [Memory Profile: {label}] ---")
        print(f" Baseline RAM   : {self.baseline_rss / (1024 * 1024):.2f} MB")
        print(f" Peak RAM       : {self.peak_mb:.2f} MB ({self.peak_gb:.3f} GB)")
        print(f" Avg RAM        : {self.avg_mb:.2f} MB")
        print(f" Net Peak       : {self.net_mb:.2f} MB ({self.net_gb:.3f} GB)")
        print(f" Net Avg        : {self.net_avg_mb:.2f} MB")
        print(f" Total Samples  : {self.samples_taken} ({self.sample_interval:.3f}s interval)")
        print("----------------------------------\n")


    def save_report(self,
                    path: str,
                    file_name: str = "memory_profile_report.txt", 
                    label: str = "Execution"
                    ) -> None:
        """Saves the memory profile report to a text file."""
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, file_name)
        report = self.summary()
        with open(filepath, 'w') as f:
            f.write(f"--- [Memory Profile: {label}] ---\n")
            for key, value in report.items():
                f.write(f"{key}: {value}\n")
            f.write("----------------------------------\n")

#> ---------------------------------------------------------------------------------------

# --- Optional Decorator for Any Function ---
def profile_memory(label: Optional[str] = None):
    """Decorator to measure peak memory during a function call."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            tag = label or func.__name__
            with MemoryTracker() as tracker:
                result = func(*args, **kwargs)
            tracker.print_report(tag)
            return result
        return wrapper
    return decorator

#> ---------------------------------------------------------------------------------------

"""
--- Usage Documentation ---

## Method 1: Wrapping .fit() directly (Context Manager)
Inside your benchmark scripts, wrap the model's fit() execution:

```python
from random_forest.forest import RandomForest
from random_forest.utils.profiler import MemoryTracker

rf = RandomForest()

# Track RAM during training
with MemoryTracker(sample_interval_sec=0.05) as tracker:
    rf.fit("X_train.npy", "Y_train.npy", n_trees=50, n_jobs=-1)

# Retrieve results directly
tracker.print_report("SAML-D Benchmark Training")

# Access programmatically for logging/benchmarking
peak_ram_used = tracker.peak_gb
print(f"Model peaked at exactly {peak_ram_used:.2f} GB of RAM")
```

#> ---------------------------------------------------------------------------------------

## Method 2: Decorating internal methods
If you want automatic memory reporting directly on your fit method in forest.py, simply decorate it:

```python
from ..utils.profiler import profile_memory

class RandomForest:
    ...
    @profile_memory(label="RF Fit Process")
    def fit(self, X_path: str, Y_path: str, ...):
        # Your existing training logic
        ...
```
"""