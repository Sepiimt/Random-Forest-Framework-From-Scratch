from typing import Optional, Literal
import numpy as np

#> ---------------------------------------------------------------------------------------

Instructions = float | Literal["auto"] | None
SplitCriterion = Literal["gini", "class_weighted_gini", "hellinger"]
Iterable = list | np.ndarray
IterableTuple = tuple[list, ...] | tuple[np.ndarray, ...]
OptionalIterable = Iterable | None
OptionalInteger = int | None
OptionalString = str | None
DependentArrayTuple = tuple[np.ndarray, np.ndarray | None]
CriteriaTuple = tuple[int | None, int | float | None, bool | None, float | None]  #> (column_number, criteria, is_numerical, score)

#> ---------------------------------------------------------------------------------------
