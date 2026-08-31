"""
## resource_limits.py
Cross-platform, cleanup-safe system resource limit decorators.

Requires:
    pip install psutil

## Public API
    @memory_limit(0.90)
    def my_function():
        ...

    @cpu_limit(0.90)
    def my_function():
        ...

    @memory_limit(0.90)
    @cpu_limit(0.90)
    def my_function():
        ...

## What This Module Does
The decorators start a lightweight daemon monitor while the decorated
function is running.

The monitor observes SYSTEM-WIDE:
    - RAM usage via psutil.virtual_memory().percent
    - CPU usage via psutil.cpu_percent(...)

It does NOT:
    - reserve RAM
    - allocate or free memory intentionally
    - throttle CPU
    - change CPU affinity
    - change process priority
    - suspend other processes
    - kill other processes
    - install or replace any process-wide OS signal handler
    - otherwise manage system resources or global interpreter state

When a limit is crossed, one of two things happens, chosen per-decorator via
`kill_switch`:

    kill_switch=False (default) - COOPERATIVE ABORT
        The specific thread running the decorated call is made to raise
        ResourceLimitExceeded the next time it executes a line of Python
        bytecode. Normal Python cleanup can happen: `finally` blocks,
        context-manager `__exit__`, ordinary exception propagation. This is
        the same trade-off every cooperative-cancellation scheme has: a
        thread stuck inside long-running native code (NumPy/BLAS/MKL/Numba/
        C extensions) will not be interrupted until it returns control to
        the interpreter.

    kill_switch=True - IMMEDIATE, UNCATCHABLE ABORT
        The monitor thread calls `os._exit()` directly, terminating the
        WHOLE PROCESS immediately at the OS level. No `finally` blocks, no
        context managers, no `atexit` hooks, no exception to catch - none of
        it runs. This does not require the target thread's cooperation at
        all, so it also fires while that thread is stuck inside native code.
        This is the "kernel-level kill switch": use it only when you truly
        want the process gone, not just the decorated call.

## Thread Compatibility
Unlike a signal-based design, this mechanism targets the specific thread
that called the decorated function - captured once, at call time, via
`threading.get_ident()`. It does not install, replace, or restore any
process-wide handler, and it does not require the main thread. Decorated
functions may be called from worker threads, thread pools, or callbacks
without restriction.

The cooperative path (`kill_switch=False`) relies on CPython's
`PyThreadState_SetAsyncExc`, a CPython-specific interpreter mechanism (not
POSIX/Windows signal delivery), so its behavior does not differ across
Windows, Linux, or macOS. It IS specific to CPython itself: on any other
Python implementation, decorating with `kill_switch=False` raises
`RuntimeError` at decoration time rather than silently no-op'ing. Use
`kill_switch=True` there instead - `os._exit()` has no such restriction.

## Multiprocessing Is Out of Scope
Both abort paths act on threads within THIS process. Neither one can reach
into, monitor, or abort a separate OS process (e.g. a `multiprocessing` or
`joblib` worker) - only the process that owns the decorated call.

## False-Positive Debouncing
`consecutive_breaches` (default 1, matching a single-sample trip) lets you
require N consecutive over-limit samples before aborting, to ride out a
brief spike from an unrelated process on the system-wide CPU reading.
"""

#> ---------------------------------------------------------------------------------------

from __future__ import annotations
import ctypes
import functools
import os
import platform
import threading
import warnings
from typing import Callable, ParamSpec, TypeVar

try:
    import psutil
except ImportError as exc:
    raise ImportError(
        "resource_limits requires the 'psutil' package. Install it with: "
        "`pip install psutil`"
    ) from exc


P = ParamSpec("P")
R = TypeVar("R")

#> ---------------------------------------------------------------------------------------

# --- Configuration ---
DEFAULT_CHECK_INTERVAL = 0.25 #> Seconds between system-wide usage samples.
_IS_CPYTHON = platform.python_implementation() == "CPython" #> `_async_raise` needs this specifically.

#> ---------------------------------------------------------------------------------------

# --- Exceptions ---
class ResourceLimitExceeded(RuntimeError):
    # --- Documentation ---
    """
    ## Exception: Resource Limit Exceeded
    Raised in the thread running the decorated call, when the cooperative
    abort path (`kill_switch=False`) fires. Never raised for `kill_switch=True`
    - that path terminates the process before any exception could be caught.

    - `resource`: `"memory"` or `"cpu"`.
    - `usage`: The observed system-wide usage, as a fraction in [0, 1+].
    - `limit`: The configured limit, as a fraction in (0, 1].
    """

    def __init__(self, resource: str, usage: float, limit: float) -> None:
        self.resource = resource
        self.usage = float(usage)
        self.limit = float(limit)
        super().__init__(
            f"System {resource} usage reached {self.usage:.1%} "
            f"(limit: {self.limit:.1%})!"
        )

#> ---------------------------------------------------------------------------------------

# --- Warnings ---
class ResourceMonitorWarning(RuntimeWarning):
    # --- Documentation ---
    """
    ## Warning: Resource Monitor
    Raised when the background monitor thread itself fails (e.g. a `psutil`
    error reading system-wide usage) and has to stop sampling early. The
    decorated function is left to keep running unmonitored rather than being
    aborted on a condition we could not actually verify.
    """

#> ---------------------------------------------------------------------------------------

# --- Validation ---
def _validate_limit(limit: float) -> float:
    if isinstance(limit, bool) or not isinstance(limit, (int, float)):
        raise TypeError("limit must be a number between `0` and `1`!")
    limit = float(limit)
    if not 0.0 < limit <= 1.0:
        raise ValueError(
            "limit must be `> 0` and `<= 1` "
            "(for example, `0.90` means 90%)!"
        )
    return limit

def _validate_interval(interval: float) -> float:
    if isinstance(interval, bool) or not isinstance(interval, (int, float)):
        raise TypeError("`check_interval` must be a positive number!")
    interval = float(interval)
    if interval <= 0.0:
        raise ValueError("`check_interval` must be `> 0`!")
    return interval

def _validate_consecutive_breaches(n: int) -> int:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("`consecutive_breaches` must be an `int`!")
    if n < 1:
        raise ValueError("`consecutive_breaches` must be `>= 1`!")
    return n

def _validate_kill_switch(flag: bool) -> bool:
    if not isinstance(flag, bool):
        raise TypeError("`kill_switch` must be a `bool`!")
    return flag

#> ---------------------------------------------------------------------------------------

# --- Call State ---
class _CallState:
    # --- Documentation ---
    """
    ## Class: Call State
    Per-call coordination between `wrapper()` and its monitor thread.

    The monitor decides whether to abort by sampling on its own schedule,
    independent of when the decorated call actually returns. Without
    coordination, a breach detected right as the call is finishing can still
    fire `PyThreadState_SetAsyncExc` a moment after `wrapper()` has already
    moved on - and once fired, that exception cannot be un-requested; it
    will land wherever the target thread happens to be executing when the
    interpreter next checks, including code entirely unrelated to this call
    (the next decorated call on that same thread).

    `lock` makes "the call has finished" and "fire the abort" mutually
    exclusive, so if the call has already finished, the monitor is
    guaranteed to see that and skip firing - the async exception never gets
    requested in the first place.
    """

    __slots__ = ("lock", "completed", "aborted")

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.completed = False #> Set once `func()` has returned or raised.
        self.aborted = False #> Set once THIS call's monitor has committed to firing.

#> ---------------------------------------------------------------------------------------

# --- Abort Mechanism ---
def _sample_usage(resource: str) -> float:
    if resource == "memory":
        return psutil.virtual_memory().percent / 100.0
    if resource == "cpu":
        return psutil.cpu_percent(interval=None) / 100.0
    raise RuntimeError(f"Unknown resource type: {resource!r}!")

def _make_async_exception_class(resource: str, usage: float, limit: float) -> type:
    #> `PyThreadState_SetAsyncExc` requires a *class*, not an instance - when
    #> the async exception actually fires, CPython instantiates it with ZERO
    #> arguments on the target thread. A plain instance is rejected outright
    #> (`SystemError: ... is not a BaseException subclass`), and passing the
    #> bare `ResourceLimitExceeded` class would fire but crash on its own
    #> `__init__`, which requires resource/usage/limit. So: build a one-off
    #> subclass per trip with those values already closed over, so zero-arg
    #> construction on the target thread still produces a fully-populated,
    #> `isinstance`-compatible `ResourceLimitExceeded`.
    def _init(self) -> None:
        ResourceLimitExceeded.__init__(self, resource, usage, limit)
    return type("ResourceLimitExceeded", (ResourceLimitExceeded,), {"__init__": _init})

def _async_raise(tid: int, exception_cls: type) -> None:
    #> CPython-specific mechanism to raise `exception_cls()` inside an
    #> arbitrary thread the next time it executes Python bytecode. This
    #> targets only the one thread running the decorated call, by its native
    #> id - it never touches process-wide signal state, so unrelated code
    #> running in other threads is left alone.
    if not (isinstance(exception_cls, type) and issubclass(exception_cls, BaseException)):
        raise TypeError("`exception_cls` Must be a `BaseException` subclass!")
    tid_c = ctypes.c_long(tid)
    affected = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_c, ctypes.py_object(exception_cls))
    if affected == 0:
        #> The target thread had already finished by the time we tried to
        #> abort it - nothing left to abort, and not an error condition.
        return
    if affected > 1:
        #> Not expected to be reachable: `PyThreadState_SetAsyncExc` is
        #> documented to affect at most one thread. Undo the call rather than
        #> risk leaving a stray pending exception on an unrelated thread, then
        #> fail loudly - this signals interpreter-level corruption, not an
        #> ordinary resource-limit condition.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid_c, None)
        raise SystemError(
            f"`PyThreadState_SetAsyncExc` Affected {affected} threads "
            f"instead of `0` or `1` - refusing to proceed!")

def _trigger_abort(
    *,
    resource: str,
    usage: float,
    limit: float,
    target_tid: int,
    kill_switch: bool,
    call_state: _CallState,
    ) -> None:
    with call_state.lock:
        if call_state.completed or call_state.aborted:
            #> `completed`: the decorated call already returned (or raised)
            #> before we got here - firing now would only land in whatever
            #> code runs next on this thread. Do nothing.
            #> `aborted`: this call's monitor already committed to firing.
            #> This does NOT guard against a second, independently-stacked
            #> decorator (e.g. `@memory_limit` over `@cpu_limit`) firing into
            #> the same thread at nearly the same time - that is an inherent
            #> property of stacking two independent aborts and is left to
            #> propagate exactly like two nested `KeyboardInterrupt`s would.
            return
        call_state.aborted = True
    if kill_switch:
        #> Immediate, uncatchable, OS-level termination of the WHOLE process.
        #> Bypasses `finally` blocks, context managers, and `atexit` entirely
        #> - and, unlike the cooperative path below, does not need the target
        #> thread to reach a Python bytecode boundary, so it also fires while
        #> that thread is stuck inside native/BLAS/Numba code.
        os._exit(1)
    exception_cls = _make_async_exception_class(resource, usage, limit)
    _async_raise(target_tid, exception_cls)

#> ---------------------------------------------------------------------------------------

# --- Monitor Worker ---
def _monitor(
    *,
    resource: str,
    limit: float,
    stop_event: threading.Event,
    check_interval: float,
    consecutive_breaches: int,
    kill_switch: bool,
    target_tid: int,
    call_state: _CallState,
    ) -> None:
    #> Lightweight monitoring thread. Exits as soon as the decorated function
    #> exits, an abort request is made, or sampling itself fails - it never
    #> waits indefinitely during function cleanup.
    if resource == "cpu":
        #> Primes psutil's non-blocking CPU counter; `interval=None` means the
        #> monitor thread does not block here.
        psutil.cpu_percent(interval=None)
    breach_streak = 0
    while not stop_event.wait(check_interval):
        try:
            usage = _sample_usage(resource)
        except psutil.Error as exc:
            #> A sampling failure is not a resource breach - stop monitoring
            #> rather than let an unhandled exception die silently inside a
            #> daemon thread, or worse, abort on a reading we never got.
            warnings.warn(
                f'"{resource}" Monitor stopped sampling after a psutil error: {exc}!',
                category=ResourceMonitorWarning,
            )
            return
        except Exception as exc:
            warnings.warn(
                f'"{resource}" Monitor stopped sampling after an unexpected error: {exc!r}!',
                category=ResourceMonitorWarning,
            )
            return
        if usage >= limit:
            breach_streak += 1
            if breach_streak >= consecutive_breaches:
                _trigger_abort(
                    resource=resource, usage=usage, limit=limit,
                    target_tid=target_tid, kill_switch=kill_switch,
                    call_state=call_state,
                )
                return
        else:
            breach_streak = 0

#> ---------------------------------------------------------------------------------------

# --- Decorators Creator ---
def _resource_limit_decorator(
    *,
    resource: str,
    limit: float,
    check_interval: float,
    consecutive_breaches: int,
    kill_switch: bool,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    limit = _validate_limit(limit)
    check_interval = _validate_interval(check_interval)
    consecutive_breaches = _validate_consecutive_breaches(consecutive_breaches)
    kill_switch = _validate_kill_switch(kill_switch)
    if not kill_switch and not _IS_CPYTHON:
        #> Fail at decoration time, not at trip time - a resource monitor
        #> that silently can't abort is worse than one that never installs.
        raise RuntimeError(
            f"'Cooperative Resource-Limit Aborts' require CPython "
            f"(current interpreter: {platform.python_implementation()}). "
            f"Pass `kill_switch=True` instead, which has no such requirement!"
        )

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # --- Starting the Monitor ---
            #> Captured once, at call time: whichever thread calls the
            #> decorated function - main or otherwise - is the abort target.
            target_tid = threading.get_ident()
            stop_event = threading.Event()
            call_state = _CallState()
            monitor = threading.Thread(
                target=_monitor,
                kwargs={
                    "resource": resource,
                    "limit": limit,
                    "stop_event": stop_event,
                    "check_interval": check_interval,
                    "consecutive_breaches": consecutive_breaches,
                    "kill_switch": kill_switch,
                    "target_tid": target_tid,
                    "call_state": call_state,
                },
                name=f"{resource}-limit-monitor",
                daemon=True,
            )
            monitor.start()
            try:
                # --- Returning ---
                return func(*args, **kwargs)
            finally:
                # --- Tearing Down the Monitor ---
                #> Unreachable if `kill_switch` fired - `os._exit()` ends the
                #> process before this line, by design.
                #>
                #> Marking `completed` FIRST, under the same lock `_trigger_abort`
                #> checks, closes the window where a breach detected right as
                #> this call finishes could still fire an async exception a
                #> moment later - one that would land in whatever code runs
                #> next on this thread, including the next decorated call.
                with call_state.lock:
                    call_state.completed = True
                stop_event.set()
                monitor.join(timeout=max(check_interval * 2.0, 0.1))
        return wrapper
    return decorator

#> ---------------------------------------------------------------------------------------

# --- Public Decorators ---
def memory_limit(
    limit: float,
    *,
    check_interval: float = DEFAULT_CHECK_INTERVAL,
    consecutive_breaches: int = 2,
    kill_switch: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # --- Documentation ---
    """
    ## Decorator: Memory Usage Limit
    Abort when SYSTEM RAM usage reaches/exceeds `limit`. Works from any
    thread, not just the main thread.\n

    :param limit: Fraction in (0, 1] - system-wide RAM usage that triggers an abort.
    :param check_interval: Seconds between usage samples.
    :param consecutive_breaches: Consecutive over-limit samples required before
        aborting. Default 2 trips on the first breaching sample; raise this to
        ride out brief spikes.
    :param kill_switch: False (default) raises `ResourceLimitExceeded` in the
        calling thread - catchable, runs `finally`/cleanup. True calls
        `os._exit()` and ends the WHOLE process immediately, uncatchably, with
        no cleanup at all - only reach for this if that is truly what you want.

    ## Usage
    - Example:
    ```python
        @memory_limit(0.90)
        def train_model():
            ...
    ```

    - Optional:
    ```python
        @memory_limit(0.90, check_interval=0.5, consecutive_breaches=3)
        def train_model():
            ...
        @memory_limit(0.90, kill_switch=True)
        def train_model():
            ...
    ```
    ## Author 
    - "Sepanta Metanat"
    """
    return _resource_limit_decorator(
        resource="memory",
        limit=limit,
        check_interval=check_interval,
        consecutive_breaches=consecutive_breaches,
        kill_switch=kill_switch,
    )


def cpu_limit(
    limit: float,
    *,
    check_interval: float = DEFAULT_CHECK_INTERVAL,
    consecutive_breaches: int = 2,
    kill_switch: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
    # --- Documentation ---
    """
    ## Decorator: CPU Usage Limit
    Abort when SYSTEM CPU usage reaches/exceeds `limit`. Works from any
    thread, not just the main thread.

    :param limit: Fraction in (0, 1] - system-wide CPU usage that triggers an abort.
    :param check_interval: Seconds between usage samples.
    :param consecutive_breaches: Consecutive over-limit samples required before
        aborting. Default 2 trips on the first breaching sample; raise this to
        ride out brief spikes from another process on the system.
    :param kill_switch: False (default) raises `ResourceLimitExceeded` in the
        calling thread - catchable, runs `finally`/cleanup. True calls
        `os._exit()` and ends the WHOLE process immediately, uncatchably, with
        no cleanup at all - only reach for this if that is truly what you want.

    ## Usage
    - Example:
    ```python
        @cpu_limit(0.90)
        def train_model():
            ...
    ```
    - Optional:
    ```python
        @cpu_limit(0.90, check_interval=0.5, consecutive_breaches=3)
        def train_model():
            ...
        @cpu_limit(0.90, kill_switch=True)
        def train_model():
            ...
    ```
    ## Author 
    - "Sepanta Metanat"
    """
    return _resource_limit_decorator(
        resource="cpu",
        limit=limit,
        check_interval=check_interval,
        consecutive_breaches=consecutive_breaches,
        kill_switch=kill_switch,
    )

#> ---------------------------------------------------------------------------------------

# --- Example / Demonstration ---
#
# from resource_limits import (
#     ResourceLimitExceeded,
#     memory_limit,
#     cpu_limit,
# )
#
#
# @memory_limit(0.90)
# def memory_job():
#     try:
#         do_expensive_work()
#     finally:
#         save_checkpoint()
#         close_files()
#         release_python_resources()
#
#
# @cpu_limit(0.90)
# def cpu_job():
#     ...
#
#
# # Both limits can be active at once.
# @memory_limit(0.90)
# @cpu_limit(0.90)
# def training():
#     ...
#
#
# # Handle the clean abort at the application boundary.
# try:
#     training()
# except ResourceLimitExceeded as exc:
#     print(f"Aborted safely: {exc}")
#
#
# # Context managers and finally blocks continue to work (kill_switch=False only):
# @memory_limit(0.90)
# def write_result():
#     with open("result.tmp", "w", encoding="utf-8") as file:
#         file.write("important data")
#         perform_work()
#
#
# # Called from a worker thread - no restriction, unlike a signal-based design:
# import threading
# t = threading.Thread(target=memory_job)
# t.start()
# t.join()
#
#
# # The nuclear option - ends the process, not just the call:
# @memory_limit(0.90, kill_switch=True)
# def unrecoverable_job():
#     ...

#> ---------------------------------------------------------------------------------------

