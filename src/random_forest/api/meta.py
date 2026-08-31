from inspect import cleandoc
from dataclasses import dataclass

#> ---------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Meta:
    # --- Optional Parameters ---
    author: str = "Sepanta Metanat"
    version: str = "1.3.0"
    github: str = "https://github.com/Sepiimt"
    # --- Technical Info Block ---
    technical_doc: str = cleandoc(f"""
    --- Technical Info ---
    1. Please be mindful of your available memory before training the model on large data. 
    \tMemory-mapping and various optimizations has been baked into the code, 
    \tbut some process may duplicate the training data (due to NumPY's advanced slicing) for a 
    \tshort time at the beggining of the process. Hence caution is advised.\n
    2. Passing `cc_indices` (categorical columns indices) through `Model.fit()` is the ONLY way 
    \tcategorical columns can be declared: X is always loaded memory-mapped, and numpy 
    \tcannot memory-map a mixed-dtype array, so there is no dtype signal left to infer 
    \tfrom by the time X reaches this function. 
    \tIf omitted, every column is treated as numerical and a warning is raised.\n
    3. On `bootstrap_balance`: Controls per-tree class balancing of the bootstrap sample.
    \tRequires a binary Y (exactly two classes). Accepts:
    \t- `None` (default): No balancing - identical to plain bootstrap (`rng.choice(n_rows, size=n_rows)`).
    \t- `float`, strictly between 0 and 1: Explicit target minority-class proportion
    \t\tper tree. `0.5` = every tree sees the minority class drawn 1:1 against the
    \t\tmajority. `0.1` = minority makes up 10% of each tree's sample (9:1 majority:minority).
    \t\tAll minority rows are drawn with replacement (size = full minority pool);
    \t\tmajority rows are drawn without replacement when enough distinct rows exist
    \t\tfor the requested count, otherwise with replacement.
    \t- `"auto"`: Target ratio is derived from the data itself as
    \t\t`min(sqrt(minority_count / total_count), 0.5)` - a moderate heuristic that
    \t\tenriches the minority class without fully equalizing it, since on extreme
    \t\tskews (e.g. 900:1) full 1:1 balancing discards most of the majority class's
    \t\tdiversity per tree. This is a heuristic default, not a canonical algorithm -
    \t\tpass an explicit float if it does not suit your data. The resolved ratio is
    \t\treported via a warning and stored on `self.bootstrap_balance_ratio`.\n
    4. On `criterion`: Split criterion, independent of and combinable with `bootstrap_balance`
    \t(one reshapes the per-tree sample, the other scores candidate splits on whatever
    \tsample it is handed). Accepts:
    \t- `"gini"` (default): Plain, unweighted Gini impurity.
    \t- `"class_weighted_gini"`: Gini impurity with samples weighted by the inverse
    \t\tfrequency of their class in the *original, whole* training `Y` - not the
    \t\tper-tree bootstrap - so it stays meaningful whether or not `bootstrap_balance`
    \t\tis also active. Requires `Y` encoded as 0/1. Resolved weights are stored on
    \t\t`self.class_weights` as `(weight_for_0, weight_for_1)`.
    \t- `"hellinger"`: Hellinger Distance between the node's class-conditional split
    \t\trates (Cieslak & Chawla, 2008) - inherently insensitive to class skew, no
    \t\tweighting involved.\n
    5. On `min_samples_split` / `min_samples_leaf`: structural regularization, independent
    \tof both `bootstrap_balance` and `criterion`.
    \t- `min_samples_split` (default `2`): a node with fewer samples than this is never
    \t\tsplit, regardless of purity.
    \t- `min_samples_leaf` (default `1`): a candidate split is rejected unless both
    \t\tresulting children would hold at least this many samples.
    \tBoth default to values that reproduce the tree's original, unregularized behaviour.\n
    """)
    # --- Usage Info Block ---
    usage_doc: str = cleandoc(f"""
    --- Usage Info ---
    1. Clean the training data and ensure x has dtypes other than `object` (e.g. int, float, bool),
    \tand `y` is a 1D array of True/False values.
    2. Save `x` and `y` as ".npy" files.
    3. Get an instance of the RandomForest class.
    4. Call upon `Model.fit()` when ready to train the model. It is important to note that `.fit()` method
    \tonly accepts the path to saved ".npy" files of `x/y`; and not `x/y` directly.\n
    - Specific info on each methods usage can be found in its docstring
    \tExample: `help(Model.fit)` \n
    """)

#> ---------------------------------------------------------------------------------------
