from __future__ import annotations
import os
import time
import threading
from typing import Optional, Dict, Any, Callable, Self
import psutil

#> ---------------------------------------------------------------------------------------

class MemoryTracker:
    # --- Documentation ---
    """
    ## Class: Memory Tracker
    Real-time process-tree memory tracker.
    
    Monitors RSS (Resident Set Size) memory across the main Python process 
    and all child processes (e.g., joblib/loky worker processes) by sampling 
    in a background thread.

    :param sample_interval_sec: How frequently (in seconds) to sample RAM.

    ## Usage
    Wrapping `.fit()` directly (Context Manager)
    Inside your benchmark scripts, wrap the model's fit() execution:

    ```python
    import MemoryTracker
    model_instance = Model()
    #> Tracks RAM during training
    with MemoryTracker(sample_interval_sec=0.05) as tracker:
        model_instance.fit(...)
    #> Retrieve results directly
    tracker.print_report("Example Benchmark")
    #> Access programmatically for logging/benchmarking
    peak_ram_used = tracker.peak_gb
    print(f"Model peaked at exactly {peak_ram_used:.2f} GB of RAM")
    ```
    ## Author 
    - "Sepanta Metanat"
    """
    # --- Methods ---
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

        # ADDED: USS (Unique Set Size) mirrors RSS above, but excludes memory
        # shared with sibling processes (e.g. mapped OpenBLAS/numpy .so pages
        # across joblib/loky workers), which RSS double-counts per process.
        self.baseline_uss: int = 0
        self.peak_uss: int = 0
        self.final_uss: int = 0
        self.sum_uss: int = 0

    def _get_process_tree_memory(self) -> tuple[int, int]:
        """
        Calculates total RSS and USS (in bytes) across parent and all active children.

        RSS (Resident Set Size) is summed per-process, which double-counts pages
        shared between processes (e.g. the same OpenBLAS/numpy .so mapped into
        several joblib/loky workers) — it overstates true physical usage as
        worker count grows. USS (Unique Set Size) excludes shared pages and is
        the physically accurate figure; it's kept alongside RSS rather than
        replacing it, since RSS is cheaper to sample and still useful as a ceiling.
        """
        total_rss = 0
        total_uss = 0
        try:
            procs = [self.parent_proc, *self.parent_proc.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            procs = [self.parent_proc]

        for proc in procs:
            try:
                if not proc.is_running():
                    continue
                full = proc.memory_full_info()
                total_rss += full.rss
                total_uss += full.uss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Process finished, terminated, or full info denied during loop
                continue
            except AttributeError:
                # USS unavailable for this process/platform; fall back to RSS only
                try:
                    total_rss += proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        return total_rss, total_uss

    def _sample_loop(self) -> None:
        """Background thread target that updates RAM metrics periodically."""
        # FIXED: Use wait() instead of time.sleep() for immediate shutdown
        while not self._stop_signal.wait(self.sample_interval):
            current_rss, current_uss = self._get_process_tree_memory()
            if current_rss > self.peak_rss:
                self.peak_rss = current_rss
            if current_uss > self.peak_uss:
                self.peak_uss = current_uss
            self.sum_rss += current_rss  # ADDED: Accumulate for average
            self.sum_uss += current_uss
            self.samples_taken += 1


    def start(self) -> Self:
        """Starts background memory profiling."""
        self.baseline_rss, self.baseline_uss = self._get_process_tree_memory()
        self.peak_rss = self.baseline_rss
        self.peak_uss = self.baseline_uss

        # FIXED: reset accumulators here, not just in __init__ — otherwise a
        # reused instance (start()/stop() called more than once) silently
        # inherits sum_rss/sum_uss/samples_taken from its previous run and
        # corrupts avg_mb / net_avg_mb going forward.
        self.sum_rss = 0
        self.sum_uss = 0
        self.samples_taken = 0

        self._stop_signal.clear()
        
        self._monitor_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._monitor_thread.start()
        return self


    def stop(self) -> Self:
        """Stops profiling and captures final RAM state."""
        self._stop_signal.set()
        if self._monitor_thread is not None:
            self._monitor_thread.join()
        
        self.final_rss, self.final_uss = self._get_process_tree_memory()
        # Final sanity check in case peak happened at the exact moment of stopping
        if self.final_rss > self.peak_rss:
            self.peak_rss = self.final_rss
        if self.final_uss > self.peak_uss:
            self.peak_uss = self.final_uss
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

    # --- ADDED: USS counterparts. These exclude memory shared between
    # processes and are the physically accurate figures when multiple
    # workers are alive at once; the *_mb/_gb properties above (RSS-based)
    # stay as-is for backward compatibility with existing call sites. ---
    @property
    def peak_uss_mb(self) -> float:
        """Peak unique (non-shared) memory used in Megabytes."""
        return self.peak_uss / (1024 * 1024)

    @property
    def peak_uss_gb(self) -> float:
        """Peak unique (non-shared) memory used in Gigabytes."""
        return self.peak_uss / (1024 * 1024 * 1024)

    @property
    def net_uss_mb(self) -> float:
        """Net increase in unique memory (Peak USS minus Baseline USS) in MB."""
        return max(0.0, (self.peak_uss - self.baseline_uss) / (1024 * 1024))

    @property
    def avg_uss_mb(self) -> float:
        """Average unique memory used in Megabytes."""
        if self.samples_taken == 0:
            return 0.0
        return (self.sum_uss / self.samples_taken) / (1024 * 1024)

    @property
    def net_avg_uss_mb(self) -> float:
        """Net average increase in unique memory (Average USS minus Baseline USS) in MB."""
        baseline_mb = self.baseline_uss / (1024 * 1024)
        return max(0.0, self.avg_uss_mb - baseline_mb)


    def summary(self) -> Dict[str, Any]:
        """Returns a clean report dictionary."""
        return {
            "baseline_mb": round(self.baseline_rss / (1024 * 1024), 2),
            "peak_mb": round(self.peak_mb, 2),
            "avg_mb": round(self.avg_mb, 2),
            "peak_gb": round(self.peak_gb, 3),
            "net_peak_mb": round(self.net_mb, 2),
            "net_avg_mb": round(self.net_avg_mb, 2),
            "peak_uss_mb": round(self.peak_uss_mb, 2),
            "avg_uss_mb": round(self.avg_uss_mb, 2),
            "peak_uss_gb": round(self.peak_uss_gb, 3),
            "net_peak_uss_mb": round(self.net_uss_mb, 2),
            "net_avg_uss_mb": round(self.net_avg_uss_mb, 2),
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
        print(f" Peak RAM (USS) : {self.peak_uss_mb:.2f} MB ({self.peak_uss_gb:.3f} GB)  [unique, excludes shared pages]")
        print(f" Avg RAM (USS)  : {self.avg_uss_mb:.2f} MB")
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
def track_memory(label: Optional[str] = None):
    # --- Documentation ---
    """
    ## Decorator: Memory Tracker
    Decorator to measure peak memory during a function call.

    ## Usage
    If you want automatic memory reporting directly on your fit method in forest.py, simply decorate it:

    - Example:
    ```python
    import profile_memory
    class Example:
        ...
        @profile_memory
        def fit(...):
            ...
    ```

    - Optional:
    ```python
    import profile_memory
    class Example:
        ...
        @profile_memory(label="Model Fit Process")
        def fit(...):
            ...
    ```
    """
    # --- Funcs ---
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
