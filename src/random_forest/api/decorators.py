from __future__ import annotations
from typing import Any, Callable, Generic, TypeVar
from functools import update_wrapper
from .exceptions import ModelNotFittedError

#> ---------------------------------------------------------------------------------------

T = TypeVar("T")
R = TypeVar("R")

class classproperty(Generic[T, R]):
    """
    ## Decorator: Class Property
    Read-only property accessible directly from the class.
    \n**Note:** Implemented as a descriptor to perform instance-aware method binding.

    ## Usage
    ```python
    class Meta:
        @classproperty
        def version(cls) -> str:
            return "1.0"

    Meta.version
    ```
    ## Author 
    - "Sepanta Metanat"
    """
    def __init__(self, func: Callable[[type[T]], R]):
        self._func = func
        self.__doc__ = getattr(func, "__doc__")

    def __get__(self, obj: Any, cls: type[T]) -> R:
        return self._func(cls)

#> ---------------------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])

class requires_fit:
    """
    ## Decorator: Fit Required
    Decorator enforcing that the instance has been
        fitted before the decorated method can be called.
    \n**Note:** Implemented as a descriptor to perform instance-aware method binding.
    ## Usage
        ```python
        class Example:
            def fit(self, ...):
                ...
                slef.is_fitted = True

            @requires_fit
            def save(self, ...):
                ...
        ```
    ## Author 
    - "Sepanta Metanat"
    """
    def __init__(self,
        func: Callable | None = None,
        *,
        message: str | None = None):
        self._func = func
        self._message = message
        update_wrapper(self, func)

    def __get__(self, obj, cls):
        if obj is None:
            return self
        def wrapper(*args, **kwargs):
            if not obj.is_fitted:
                if self._message is None:
                    raise ModelNotFittedError(
                        f"Cannot call `.{self._func.__name__}()` before calling `.fit()`!")
                else:
                    raise ModelNotFittedError(self._message)
            return self._func(obj, *args, **kwargs)
        update_wrapper(wrapper, self._func)
        return wrapper

#> ---------------------------------------------------------------------------------------
