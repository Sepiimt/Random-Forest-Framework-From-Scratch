import numpy as np
from typeguard import typechecked
from ..api.typing import Iterable
#> ---------------------------------------------------------------------------------------

@typechecked
def train_test_split(X: Iterable, 
                     Y: Iterable,
                     *,
                     test_size: float = 0.2, 
                     subsample: float | int | None = None,
                     shuffle: bool = True, 
                     random_state: int | None = None
                     ) -> tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    # --- Documentation ---
    """\u200b
    --- Train/Test Split ---
    Function implemented from scratch by "Sepanta Metanat"

    First edit: "2026/02/25"
    Last edit: "2026/08/5"

    #> Usage and Information
    Use this function to split train/test data while customizing it.
        
    #> Parameters Documentation: 
    1. X: X array
    2. Y: Y array
    3. test_size: Based on total size, the amount dedicated to testing
    4. subsample: Percentage or sample-count to create subsample from total
    5. shuffle: Flag for shuffling the data
    6. random_state: Random state value
    \t(to be able to encode new data based on training)
    """
    X = np.asarray(X)
    Y = np.asarray(Y).ravel()
    # --- Checking for same length ---
    if len(X) != len(Y):
        raise ValueError("Error: X and Y must have the same number of samples!")
    n_samples = len(X)
    indices = np.arange(n_samples)
    # --- If Shuffle is True ---
    if shuffle:
        shuffle_rng = np.random.default_rng(random_state)
        shuffle_rng.shuffle(indices)
    # --- Subsampling ---
    if subsample is not None:
        if isinstance(subsample, float):
            if not (0.0 < subsample <= 1.0):
                raise ValueError("subsample float must be in range (0.0, 1.0]")
            n_subsample = int(n_samples * subsample)
        else:
            if not (0 < subsample <= n_samples):
                raise ValueError(f"subsample int must be between 1 and {n_samples}")
            n_subsample = subsample
        
        indices = indices[:n_subsample]
        n_samples = len(indices)
    # --- Calculating Test-Array Size ---
    n_test = int(n_samples * test_size)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    # --- Making the Final Arrays Ready ---
    x_train = X[train_idx]
    x_test  = X[test_idx]
    y_train = Y[train_idx]
    y_test  = Y[test_idx]
    # --- Return ---
    return x_train, x_test, y_train, y_test

#> ---------------------------------------------------------------------------------------

"""
--- Usage Examples ---

```python

# Select 30% of total data, then split 80/20 train/test
x_tr, x_te, y_tr, y_te = train_test_split(X, Y, subsample=0.30, test_size=0.2)

# Select exactly 4000 samples of total data, then split 80/20 train/test
x_tr, x_te, y_tr, y_te = train_test_split(X, Y, subsample=4000, test_size=0.2)

```
"""

#> ---------------------------------------------------------------------------------------
