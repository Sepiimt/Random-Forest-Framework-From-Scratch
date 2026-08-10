# --- Errors ---

class RandomForestError(Exception):
    """Base exception for every RandomForest error."""

class MemoryLimitExceededError(RandomForestError):
    """Process exceeded its allowed RAM limit!"""

class ModelNotFittedError(RandomForestError):
    """Prediction attempted before training!"""

class AlreadyFittedError(RandomForestError):
    """Attempted to fit an already-trained model! To force refitting, set `force_refit=True` in the `fit()` method."""

class InvalidDimensionError(RandomForestError):
    """'Provided `X` or `Y` has invalid dimensions!'"""

class InvalidInputError(RandomForestError):
    """Invalid input provided!"""

class BootstrapBalanceError(RandomForestError):
    """Invalid bootstrap balancing configuration!"""

class SerializationError(RandomForestError):
    """Saving/loading failed!"""

class InvalidPathError(RandomForestError):
    """'Provided path for `X` or `Y` data is not valid!'"""