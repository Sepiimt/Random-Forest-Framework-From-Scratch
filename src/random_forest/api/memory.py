import functools
import multiprocessing as mp
import psutil
from .exceptions import MemoryLimitExceededError

#> ---------------------------------------------------------------------------------------

def _worker_runner(func, queue, args, kwargs):
    try:
        result = func(*args, **kwargs)
        queue.put((True, result))
    except Exception as e:
        queue.put((False, e))

#> ---------------------------------------------------------------------------------------

def limit_memory(
        ram_fraction: float, 
        check_interval_sec: float = 0.02):
    # --- Documentation ---
    """
    ## Info & Usage
    Decorator that limits a process tree's RAM usage to a percentage of total physical RAM.
    - Usage: `@limit_memory(0.95)`

    :param ram_fraction: Max allowed RAM as a fraction between 0.0 and 1.0 (e.g., 0.98 for 98%).
    :param check_interval_sec: Polling interval in seconds.
    """
    if not (0.0 < ram_fraction <= 1.0):
        raise ValueError("ram_fraction must be a float between 0.0 and 1.0 (e.g., 0.98).")
    # --- Decorator ---
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # --- Getting Info ---
            total_ram_bytes = psutil.virtual_memory().total
            max_bytes = int(total_ram_bytes * ram_fraction)
            ctx = mp.get_context("spawn")
            queue = ctx.Queue()
            proc = ctx.Process(target=_worker_runner, args=(func, queue, args, kwargs))
            proc.start()
            p_psutil = None
            try:
                p_psutil = psutil.Process(proc.pid)
            except psutil.NoSuchProcess:
                pass
            # --- Profiler Loop ---
            while proc.is_alive():
                if p_psutil is not None:
                    try:
                        total_rss = p_psutil.memory_info().rss
                        for child in p_psutil.children(recursive=True):
                            try:
                                total_rss += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        if total_rss > max_bytes:
                            for child in p_psutil.children(recursive=True):
                                try:
                                    child.kill()
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                            proc.kill()
                            proc.join()
                            # --- Raising Error ---
                            used_gb = total_rss / (1024 ** 3)
                            max_gb = max_bytes / (1024 ** 3)
                            used_pct = (total_rss / total_ram_bytes) * 100
                            raise MemoryLimitExceededError(
                                f"Process terminated! Used {used_gb:.2f} GB ({used_pct:.1f}% of total RAM). "
                                f"Limit: {ram_fraction * 100:.1f}% ({max_gb:.2f} GB)."
                            )
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                proc.join(timeout=check_interval_sec)
            # --- Empty ---
            if not queue.empty():
                success, payload = queue.get()
                if success:
                    return payload
                raise payload
            elif proc.exitcode != 0 and proc.exitcode is not None:
                raise RuntimeError(f"Subprocess terminated unexpectedly with exit code {proc.exitcode}")
        # --- Return ---
        return wrapper
    return decorator

#> ---------------------------------------------------------------------------------------
