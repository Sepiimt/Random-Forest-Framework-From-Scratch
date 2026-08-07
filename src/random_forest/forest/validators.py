import os
import numpy as np
from ..api.exceptions import InvalidPathError, AlreadyFittedError, InvalidDimensionError
from ..api.typing import OptionalIterable, OptionalString

#> ---------------------------------------------------------------------------------------

def _refit_validation(forest: object, 
                      force_refit: bool
                      ) -> None:
    if forest.is_fitted:
        if force_refit:
            return
        else:
            raise AlreadyFittedError

#> ---------------------------------------------------------------------------------------

def _input_path_validator(X_path: str, 
                          Y_path: OptionalString = None
                          ) -> None:
    # --- X Validation ---
    X_status = os.path.exists(X_path)
    if not X_status:
            raise InvalidPathError
    # --- if Y is provided ---
    if Y_path is not None:
        Y_status = os.path.exists(Y_path)
        if not Y_status:
                raise InvalidPathError

#> ---------------------------------------------------------------------------------------

def _input_validator(X: np.ndarray, 
                     Y: np.ndarray | None = None
                     ) -> None:
    # --- Checking the Inputs Type ---
    if not isinstance(X, np.ndarray):
        raise TypeError('Please provide `X` as `numpy.ndarray`!')
    # --- Checking the Inputs Logic ---
    if X.ndim == 1:
        raise InvalidDimensionError('Provided `X` has no more than 1 dimension (is a vector)!')
    # --- if Y is provided ---
    if Y is not None:
        # --- Checking the Inputs Type ---
        if not isinstance(Y, np.ndarray):
            raise TypeError('Please provide `Y` as `numpy.ndarray`!')    
        # --- Checking the Inputs Logic ---
        if Y.ndim != 1:
            raise InvalidDimensionError('Provided `Y` has more than 1 dimension!')

#> ---------------------------------------------------------------------------------------