# --- Warnings ---

class RandomForestWarning(UserWarning):
    """Base warning for every RandomForest warning!"""

class CategoricalInferenceWarning(RandomForestWarning):
    """No cc_indices (Categorical Column Indices) were provided!"""

class ResourceLimitWarning(RandomForestWarning):
    """Current task may overload CPU capacity!"""

class AutoBalanceWarning(RandomForestWarning):
    """bootstrap_balance='auto' resolved a ratio!"""