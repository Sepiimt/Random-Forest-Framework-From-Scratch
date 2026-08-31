# --- Errors ---

class RandomForestError(Exception):
    """
    ## Exception: RandomForest Error
    Base exception for every RandomForest error.
    """

class ModelNotFittedError(RandomForestError):
    """
    ## Exception: Model Not Fitted Error
    Prohibited action attempted before training the model!
    """

class AlreadyFittedError(RandomForestError):
    """
    ## Exception: Already Fitted Error
    Attempted to fit an already-trained model! To force refitting, set `force_refit=True` in the `fit()` method.
    """

class InvalidDimensionError(RandomForestError):
    """
    ## Exception: Invalid Dimension Error
    Provided `X` or `Y` has invalid dimensions!
    """

class InvalidInputError(RandomForestError):
    """
    ## Exception: Invalid Input Error
    Invalid input provided!
    """

class BootstrapBalanceError(RandomForestError):
    """
    ## Exception: Bootstrap Balance Error
    Invalid bootstrap balancing configuration!
    """

class SerializationError(RandomForestError):
    """
    ## Exception: Serialization Error
    Saving/loading failed!
    """

class InvalidPathError(RandomForestError):
    """
    ## Exception: Invalid Path Error
    Provided path for `X` or `Y` data is not valid!
    """