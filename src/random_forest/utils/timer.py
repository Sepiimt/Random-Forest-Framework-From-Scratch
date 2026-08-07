from contextlib import contextmanager
from time import perf_counter

#> ---------------------------------------------------------------------------------------

@contextmanager
def timer_function(
    name: str = "Operation", 
    enabled: bool = True
    ):
    start = perf_counter()
    try:
        yield
    finally:
        if enabled:
            elapsed = perf_counter() - start
            print(f"{name} Elapsed Time: {elapsed:.4f} Seconds.")

#> ---------------------------------------------------------------------------------------

def time_capture_function(status: str) -> float | None:
        if status == "start":
            global temp_time_stamp
            temp_time_stamp = perf_counter()
        if status == "end":
            elapsed = perf_counter() - temp_time_stamp
            del temp_time_stamp
            return elapsed
        
#> ---------------------------------------------------------------------------------------