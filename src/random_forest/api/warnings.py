# --- Warnings ---

class RandomForestWarning(UserWarning):
    """
    ## Warning: RandomForest Warning
    Base warning for every RandomForest warning!
    """

class CategoricalInferenceWarning(RandomForestWarning):
    """
    ## Warning: Categorical Inference Warning
    No cc_indices (Categorical Column Indices) were provided!
    """

class ResourceLimitWarning(RandomForestWarning):
    """
    ## Warning: Resource Limit Warning
    Current task may overload CPU capacity!
    """

class AutoBalanceWarning(RandomForestWarning):
    """
    ## Warning: Auto-Balance Warning
    `bootstrap_balance='auto'` resolved a ratio!
    """