import gc
import numpy as np
from numba import njit
import os
import warnings
from typing import Optional, Self
from collections.abc import Generator
from inspect import cleandoc
from typeguard import typechecked
from joblib import Parallel, delayed
from .validators import _refit_validation, _input_path_validator, _input_validator
from ..utils import timer_function, time_capture_function
from ..tree import Node
from ..api.meta import Meta as meta
from ..api.decorators import classproperty, requires_fit
from ..api.resource import memory_limit, cpu_limit

from ..api.warnings import (CategoricalInferenceWarning, 
                            ResourceLimitWarning, AutoBalanceWarning)

from ..api.typing import (Instructions, SplitCriterion, Iterable, IterableTuple, 
                          OptionalIterable, DependentArrayTuple, OptionalInteger)

#> ---------------------------------------------------------------------------------------

class RandomForest:
    # --- Documentation ---
    """
    ## Class: Random Forest
    RF framework implemented from scratch by "Sepanta Metanat"
    - First edit: "2026/02/25"
    - Last edit: "2026/08/30"
    """
    # --- Slots ---
    __slots__ = (
        "tree_list", "n_trees", "n_features", "n_samples", "tree_depth", 
        "trees_node_min_purity", "train_time_taken", "bootstrap_balance", 
        "bootstrap_balance_ratio","criterion", "class_weights", 
        "min_samples_split", "min_samples_leaf", "is_fitted")
    
    def __init__(self):
        # --- Training Related ---
        self.tree_list = None #> a list or an array to store Tree Objects.
        self.n_trees = None #> How many trees shall be created from data.
        self.n_features = None #> How many columns did our x array had.
        self.n_samples = None #> How many samples have been provided for training.
        self.tree_depth = None #> Each Tree's depth.
        self.trees_node_min_purity = None #> Minimum leaf node purity.
        self.train_time_taken = None #> Training time taken.
        self.bootstrap_balance = None #> As passed to `.fit()`: None / float ratio / "auto".
        self.bootstrap_balance_ratio = None #> Resolved minority-class target ratio actually used (None if bootstrap_balance was None).
        self.criterion = None #> Split criterion: "gini" / "class_weighted_gini" / "hellinger".
        self.class_weights = None #> Resolved (weight_for_0, weight_for_1), only set when criterion == "class_weighted_gini".
        self.min_samples_split = None #> Structural regularization: minimum samples a node needs to attempt a split.
        self.min_samples_leaf = None #> Structural regularization: minimum samples required on both sides of a candidate split.
        # --- Flags ---
        self.is_fitted: bool = False #> Is trained or not.
    
    @classproperty
    def info(cls) -> str:
        return cleandoc(f"Please refer to the `RandomForest.meta` class for usage information and details.")

    @classproperty
    def meta(cls) -> meta:
        # --- Meta Class ---
        return meta()


    @cpu_limit(0.9)
    @memory_limit(0.9)
    @typechecked
    def fit(self, 
            X_path: str, 
            Y_path: str,
            *,
            cc_indices: OptionalIterable = None,
            n_trees: int = 5, 
            trees_max_depth: int = 5, 
            min_leaf_purity: float = 0.95,
            bootstrap_balance: Instructions = None,
            criterion: SplitCriterion = "gini",
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            random_state: int = 42,
            n_jobs: OptionalInteger = None,
            force_refit: bool = False,
            timer: bool = True
            ) -> None:
        # --- Documentation ---
        """
        ## Usage
        Use this function to train RF model.
             
        :param X_path: Training X `.npy` file path.
        :param Y_path: Training Y `.npy` file path.
        :param cc_indices: Column indices (into X) that are categorical.
        :param n_trees: Number of Trees you desire to train in RF model.
        :param trees_max_depth: Each tree's maximum allowed depth.
        :param min_leaf_purity: Node's minimum purity to turn leaf.
        :param bootstrap_balance: Controls per-tree class balancing of the bootstrap sample.
        :param criterion: Split criterion, independent of and combinable with `bootstrap_balance`.
        :param min_samples_split: Structural regularization - a node with fewer samples than this 
        \tis never split, regardless of purity.
        :param min_samples_leaf: Structural regularization - a candidate split is rejected unless 
        \tboth children would end up with at least this many samples.
        :param random_state: The number of random state.
        :param n_jobs: Number of CPU cores you desire to include in parallel computing.
        :param force_refit: True/False Flag for forcing a trained model to refit.
        :param timer: True/False Flag for timer. 
        """
        # --- Input Validation ---
        _refit_validation(self, force_refit)
        _input_path_validator(X_path, Y_path=Y_path)
        self._structural_regularization_validator(min_samples_split, 
                                                  min_samples_leaf, 
                                                  min_leaf_purity)
        n_processes = self._process_amount_determiner(n_jobs)
        X, Y = self._input_memory_mapper(X_path, Y_path=Y_path)
        _input_validator(X, Y=Y)
        # --- Storing Information ---
        self._value_reset(X, n_trees, trees_max_depth, min_leaf_purity, bootstrap_balance,
                           criterion, min_samples_split, min_samples_leaf)
        is_numerical_mask = self._build_numerical_mask(X, cc_indices)
        # --- Preparing for Training ---
        numerical_column_indices, column_position_map = self._build_column_position_map(is_numerical_mask)
        X_numerical = X[:, numerical_column_indices].astype(np.float32) if numerical_column_indices.size > 0 else np.empty((len(X), 0))
        balance_info, self.bootstrap_balance_ratio = self._bootstrap_balance_resolver(Y, bootstrap_balance)
        self.class_weights = self._class_weights_resolver(Y, criterion)
        # --- Initelizing Related Tools ---
        master = np.random.SeedSequence(random_state)
        child_sequences = master.spawn(n_trees)
        # --- Time Value Related ---
        time_capture_function("start")
        # --- Creating Parallel Processes and Training The Trees ---
        with timer_function("RF Model Training", timer):
            self.tree_list = Parallel(n_jobs=n_processes, 
                                    backend="loky",
                                    max_nbytes="1M")(
                                        delayed(self._create_decision_tree)(
                                            X, Y, timer, 
                                            trees_max_depth, min_leaf_purity, 
                                            is_numerical_mask,
                                            X_numerical,
                                            column_position_map,
                                            balance_info,
                                            criterion, self.class_weights,
                                            min_samples_split, min_samples_leaf,
                                            np.random.default_rng(child_sequences[i])
                                            )for i in range(n_trees))
        # --- Releasing Fit-Only State ---
        #> Note: `is_numerical_mask` and `column_position_map` exist purely to route
        #> this training loop - every `Node` already carries its own resolved
        #> `is_numerical`/`column_number` after splitting, and `predict()` never
        #> reads either off `self`. Holding them past this point only inflates what
        #> `joblib.dump` has to walk and what a fitted instance keeps resident.
        del X_numerical, is_numerical_mask, column_position_map
        gc.collect()
        # --- Saving Time Value ---
        self.train_time_taken = time_capture_function("end")
        # --- Chagning the Flag ---
        self.is_fitted = True

    @staticmethod
    def _input_memory_mapper(
        X_path: str, 
        Y_path: str | None = None
        ) -> DependentArrayTuple:
        # --- Memory Mapping ---
        X = np.load(X_path, mmap_mode='r')
        Y = np.load(Y_path, mmap_mode='r') if Y_path is not None else None
        return X, Y

    @staticmethod
    def _build_numerical_mask(
        X: Iterable,
        categorical_columns: OptionalIterable
        ) -> Iterable:
        n_columns = X.shape[1]
        # --- If cc_columns in Parsed ---
        if categorical_columns is not None:
            cat_arr = np.asarray(categorical_columns, dtype=np.int32)
            if cat_arr.size and (cat_arr.max() >= n_columns or cat_arr.min() < 0):
                raise ValueError('Provided `cc_indices` contains an out-of-range column index!')
            mask = np.ones(n_columns, dtype=bool)
            mask[cat_arr] = False
            return mask
        # --- Else ---
        else:
            warnings.warn(
                'No `cc_indices` (categorical_columns) provided: every column will be treated as numerical. '
                'Memory-mapped X cannot carry mixed dtypes, so this cannot be auto-detected - '
                'pass `cc_indices` explicitly if any feature is a pre-encoded category.',
                category=CategoricalInferenceWarning)
            return np.ones(n_columns, dtype=bool)

    @staticmethod
    @njit
    def _build_column_position_map(is_numerical_mask) -> IterableTuple:
        numerical_column_indices = np.where(is_numerical_mask)[0]
        column_position_map = np.full(is_numerical_mask.shape[0], -1, dtype=np.int32)
        column_position_map[numerical_column_indices] = np.arange(numerical_column_indices.size)
        return numerical_column_indices, column_position_map

    @staticmethod
    def _bootstrap_balance_resolver(Y,
                                    bootstrap_balance
                                    ) -> tuple[Optional[tuple], Optional[float]]:
        # --- Off: Return Immediately, Preserving the Original Unweighted Bootstrap ---
        if bootstrap_balance is None:
            return None, None
        # --- This Feature Only Makes Sense for Binary Y (Matches the Rest of the Codebase) ---
        unique_labels, counts = np.unique(Y, return_counts=True)
        if unique_labels.size != 2:
            raise ValueError('`bootstrap_balance` requires a binary `Y` (exactly two classes)!')
        minority_label = unique_labels[np.argmin(counts)]
        majority_label = unique_labels[np.argmax(counts)]
        minority_indices = np.where(Y == minority_label)[0]
        majority_indices = np.where(Y == majority_label)[0]
        minority_count = minority_indices.size
        majority_count = majority_indices.size
        # --- Resolving the Requested Ratio into a Concrete Target ---
        if isinstance(bootstrap_balance, str):
            if bootstrap_balance.lower() != "auto":
                raise ValueError('`bootstrap_balance` only supports `"auto"` as a string value!')
            ratio = RandomForest._auto_balance_ratio(minority_count, majority_count)
            warnings.warn(f"""`bootstrap_balance="auto"` resolved to a minority ratio of {ratio:.4%} 
            per tree (~1:{(1-ratio)/ratio:.1f} majority:minority).""", category=AutoBalanceWarning)
        elif isinstance(bootstrap_balance, float):
            if not (0.0 < bootstrap_balance < 1.0):
                raise ValueError('`bootstrap_balance` ratio must be strictly between 0 and 1!')
            ratio = bootstrap_balance
        else:
            raise TypeError('`bootstrap_balance` must be `None`, a `float` ratio, or `"auto"`!')
        # --- Translating the Ratio into How Many Majority Rows Each Tree Draws ---
        majority_draw_count = max(1, round(minority_count * (1 - ratio) / ratio))
        return (minority_indices, majority_indices, majority_draw_count), ratio

    @staticmethod
    def _auto_balance_ratio(minority_count, majority_count) -> float:
        #> Note: Dampens the *true* imbalance with a square root rather than fully
        #> equalizing to 50/50. On extreme skews (e.g. ~962:1, as in SAML-D) full 1:1
        #> balancing would throw away nearly all majority-class diversity in every tree.
        #> This is a deliberately moderate heuristic, not a canonical named algorithm -
        #> pass an explicit float to `bootstrap_balance` to override it.
        true_ratio = minority_count / (minority_count + majority_count)
        return float(min(np.sqrt(true_ratio), 0.5))

    @staticmethod
    def _class_weights_resolver(Y, criterion) -> Optional[tuple]:
        # --- Only "class_weighted_gini" Needs Weights ---
        if criterion != "class_weighted_gini":
            return None
        # --- Resolved from the Original, Whole `Y` - Deliberately NOT the Per-Tree Bootstrap ---
        #> Note: this is what keeps the criterion independent of, and freely combinable
        #> with, `bootstrap_balance` (which only ever reshapes the per-tree sample). Mirrors
        #> sklearn's "balanced" (whole-dataset) heuristic rather than "balanced_subsample".
        unique_labels, counts = np.unique(Y, return_counts=True)
        if unique_labels.size != 2 or not np.array_equal(unique_labels, np.array([0, 1])):
            raise ValueError('`criterion="class_weighted_gini"` requires `Y` encoded as 0/1 '
                              '(exactly two classes, matching the rest of the codebase)!')
        n_total = counts.sum()
        weight_0, weight_1 = n_total / (2.0 * counts)
        return float(weight_0), float(weight_1)

    @staticmethod
    def _structural_regularization_validator(min_samples_split: int, 
                                             min_samples_leaf: int, 
                                             min_leaf_purity: float
                                             ) -> None:
        if min_samples_split < 2:
            raise ValueError('`min_samples_split` must be >= 2!')
        if min_samples_leaf < 1:
            raise ValueError('`min_samples_leaf` must be >= 1!')
        if min_leaf_purity < 0 or min_leaf_purity > 1:
            raise ValueError('`min_leaf_purity` must be between 0 and 1!')

    @staticmethod
    def _process_amount_determiner(n_jobs: int) -> int:
        cpu_cnt = os.cpu_count() or 1
        # --- Handling None/Negative Values ---
        if n_jobs is None:
            return max(1, cpu_cnt // 2)
        elif n_jobs == -1:
            return max(1, cpu_cnt - 1)
        # --- CPU Compatibility Check ---
        elif isinstance(n_jobs, int) and n_jobs > 0:
            if n_jobs > cpu_cnt//2:
                warnings.warn("Provided `n_jobs` may exceed CPU's physical cores!", 
                              category=ResourceLimitWarning)
            return n_jobs
        else:
            raise ValueError('Provided `n_jobs` must be `None`, -1, or a positive integer!')
    
    def _value_reset(self, X, n_trees, trees_max_depth, min_leaf_purity, 
                     bootstrap_balance, criterion, min_samples_split, 
                     min_samples_leaf) -> None:
        self.n_samples = len(X)
        self.n_features = X.shape[1]
        self.n_trees = n_trees
        self.tree_depth = trees_max_depth
        self.trees_node_min_purity = min_leaf_purity
        self.tree_list = []
        self.bootstrap_balance = bootstrap_balance
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _create_decision_tree(self, 
                              X: Iterable, 
                              Y: Iterable, 
                              timer: bool, 
                              trees_max_depth: int, 
                              min_leaf_purity: float, 
                              is_numerical_mask: Iterable, 
                              X_numerical: Iterable, 
                              column_position_map: dict, 
                              balance_info: tuple, 
                              criterion: str, 
                              class_weights, 
                              min_samples_split: int, 
                              min_samples_leaf: int, 
                              rng: Generator
                              ) -> Node:
        with timer_function("Tree Training", timer):
            # --- Creating Bootstraped Indices ---
            #> Note: balance_info is None unless `bootstrap_balance` was passed to `.fit()`,
            #> in which case it is (minority_indices, majority_indices, majority_draw_count),
            #> resolved once in `.fit()` rather than per tree.
            if balance_info is None:
                random_row_indices = self._data_bootstrapper(len(Y), rng)
            else:
                random_row_indices = self._balanced_data_bootstrapper(rng, *balance_info)
            # --- Sort x and align y ---
            #> Note: It is very important to sort our data, as we will be calculating criteria at the end 
            #> and we need to calculate ginis the way we could know which gini belongs 
            #> to which criteria. We do it here once to preserve performance.
            #> Note: X_numerical already excludes categorical columns AND was already
            #> sliced out of the full X once in fit() - here we only pay for the
            #> per-tree row-bootstrap gather, not a repeated column selection too.
            sort_order_array = np.argsort(X_numerical[random_row_indices], axis=0).astype(np.int32)
            # --- Creating the Tree ---
            Tree = Node()
            Tree.fit(X, Y, random_row_indices, sort_order_array, 
                     trees_max_depth, min_leaf_purity, is_numerical_mask, 
                     column_position_map, criterion, class_weights, 
                     min_samples_split, min_samples_leaf, rng)
        # --- Returning ---
        return Tree

    @staticmethod
    def _data_bootstrapper(n_rows: int, rng: Generator) -> Iterable:
        # --- Choosing Random Rows ---
        random_row_indices = rng.choice(n_rows, size=n_rows)
        # --- Returning ---
        return random_row_indices

    @staticmethod
    def _balanced_data_bootstrapper(rng: Generator, 
                                    minority_indices, 
                                    majority_indices, 
                                    majority_draw_count
                                    ) -> Iterable:
        # --- Minority Side: Always Drawn With Replacement ---
        #> Note: the minority pool is small and reused by every tree, so replacement
        #> is what lets different trees see different emphasis/duplication of it.
        minority_draw = rng.choice(minority_indices, size=minority_indices.size, replace=True)
        # --- Majority Side: Without Replacement Whenever the Pool Allows It ---
        #> Note: sampling without replacement here maximizes per-tree diversity across
        #> the (typically enormous) majority pool; falls back to replacement only if
        #> more draws were requested than distinct majority rows exist.
        replace_majority = majority_draw_count > majority_indices.size
        majority_draw = rng.choice(majority_indices, size=majority_draw_count, replace=replace_majority)
        # --- Combine and Shuffle so Downstream Code Sees no Class-Block Ordering ---
        combined_row_indices = np.concatenate([minority_draw, majority_draw])
        rng.shuffle(combined_row_indices)
        # --- Returning ---
        return combined_row_indices


    @requires_fit #> Model Train Check
    @cpu_limit(0.9)
    @memory_limit(0.9)
    @typechecked
    def predict(self, 
                X_path: str,
                *, 
                binarize: bool = False,
                threshold: float = 0.5,
                n_jobs: OptionalInteger = None,
                timer: bool = True
                ) -> Iterable:
        # --- Documentation ---
        """
        ## Usage
        Use this function to use the trained RF model and predict.

        :param X_path: Testing X `.npy` file path.
        :param binarize: Flag to determine if predictions should be binarized.
        :param threshold: Threshold for binarizing predictions.
        :param timer: Flag for starting the timer.
        """
        # --- Input Validation ---
        _input_path_validator(X_path)
        X, _ = self._input_memory_mapper(X_path)
        _input_validator(X)
        self._train_test_data_match(X.shape[1])
        n_processes = self._process_amount_determiner(n_jobs)
        # --- if Detailed_pred ---
        if binarize:
            return self._binarize(self._predict_process(X, n_processes, timer), threshold)
        else:
            return self._predict_process(X, n_processes, timer)

    def _train_test_data_match(self, n_columns: int) -> None:
        if n_columns != self.n_features:
            raise TypeError(f"Entered X's features does not match the trained!")

    def _predict_process(self, X, n_processes, timer) -> Iterable:
        # --- Deciding Over Multi-Processing ---
        with timer_function("Predicting Process", timer):
            if n_processes == 1 or len(X) < 1000:
                tree_preds = [Tree.predict(X) for Tree in self.tree_list]
            else:
                tree_preds = Parallel(n_jobs=n_processes, backend="loky", max_nbytes="1M")(
                    delayed(tree.predict)(X) for tree in self.tree_list)
        # -- Averaging Probabilities Across All Trees ---
        predictions = np.mean(tree_preds, axis=0)
        # --- Return ---
        return predictions

    @staticmethod
    def _binarize(predicted_y: np.ndarray, threshold: float) -> Iterable:
        return (predicted_y >= threshold).astype(int)


    @requires_fit #> Model Train Check
    @cpu_limit(0.9)
    @memory_limit(0.9)
    @typechecked
    def save_model(self, 
                   directory: str = r"../artifacts/forest/", 
                   silent_save: bool = False
                   ) -> None:
        """
        ## Usage
        Use this function to save the RF model.
            
        :param directory: Path to desired directory which the model will be saved in.
        :param silent_save: Silent's the "Successful Saving" message.
        """
        # --- Creating Directory and Path ---
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, "random_forest_model.npz")
        # --- Flattening Trees Into Arrays + Packing Scalar Metadata ---
        tree_arrays = self._flatten_trees(self.tree_list)
        meta_arrays = self._pack_metadata()
        # --- Saving ---
        np.savez(filepath, **tree_arrays, **meta_arrays)
        # --- Printing info ---
        if not silent_save:
            print(f"Model Successfully Has Been Saved!")

    @staticmethod
    def _flatten_trees(tree_list: list) -> dict:
        #> Flattens the recursive Node tree(s) into parallel numpy arrays
        #> (preorder), so `left_child`/`right_child` become integer indices
        #> into these arrays instead of nested Python object references.
        is_leaf, is_numerical = [], []
        criteria, node_score = [], []
        column_number = []
        left_child, right_child = [], []
        prediction, probability = [], []
        def _flatten(node: Node) -> int:
            idx = len(is_leaf)
            is_leaf.append(bool(node.is_leaf))
            is_numerical.append(bool(node.is_numerical) if node.is_numerical is not None else False)
            criteria.append(node.criteria if node.criteria is not None else np.nan)
            node_score.append(node.node_score if node.node_score is not None else np.nan)
            column_number.append(node.column_number if node.column_number is not None else -1)
            prediction.append(node.prediction if node.prediction is not None else np.nan)
            probability.append(node.probability if node.probability is not None else np.nan)
            left_child.append(-1)
            right_child.append(-1)
            if not node.is_leaf:
                left_child[idx] = _flatten(node.left_child)
                right_child[idx] = _flatten(node.right_child)
            return idx
        tree_offsets = np.empty(len(tree_list) + 1, dtype=np.int32)
        tree_offsets[0] = 0
        for i, root in enumerate(tree_list):
            _flatten(root)
            tree_offsets[i + 1] = len(is_leaf)

        #> Note: criteria/prediction/probability stay float64 - float32 silently
        #> corrupts pre-encoded categorical codes above ~16.7M, which is a real
        #> risk on a financial transaction dataset (account/merchant IDs, etc.).
        return {
            "node_is_leaf": np.asarray(is_leaf, dtype=bool),
            "node_is_numerical": np.asarray(is_numerical, dtype=bool),
            "node_criteria": np.asarray(criteria, dtype=np.float64),
            "node_score": np.asarray(node_score, dtype=np.float64),
            "node_column": np.asarray(column_number, dtype=np.int32),
            "node_left": np.asarray(left_child, dtype=np.int32),
            "node_right": np.asarray(right_child, dtype=np.int32),
            "node_prediction": np.asarray(prediction, dtype=np.float64),
            "node_probability": np.asarray(probability, dtype=np.float64),
            "tree_offsets": tree_offsets,
        }
    
    def _pack_metadata(self) -> dict:
        #> None-safe scalar packing - np.savez can't hold Python `None` directly,
        #> so Optional fields get a companion "kind"/NaN-sentinel encoding.
        if self.bootstrap_balance is None:
            bb_kind, bb_value = "none", np.nan
        elif isinstance(self.bootstrap_balance, str):
            bb_kind, bb_value = "auto", np.nan
        else:
            bb_kind, bb_value = "float", float(self.bootstrap_balance)
        class_weights = (np.array(self.class_weights, dtype=np.float64)
                          if self.class_weights is not None else np.array([np.nan, np.nan]))
        return {
            "n_trees": np.int32(self.n_trees),
            "n_features": np.int32(self.n_features),
            "n_samples": np.int64(self.n_samples),
            "tree_depth": np.int32(self.tree_depth),
            "trees_node_min_purity": np.float64(self.trees_node_min_purity),
            "train_time_taken": np.float64(self.train_time_taken) if self.train_time_taken is not None else np.nan,
            "bootstrap_balance_kind": np.str_(bb_kind),
            "bootstrap_balance_value": np.float64(bb_value),
            "bootstrap_balance_ratio": (np.float64(self.bootstrap_balance_ratio)
                                         if self.bootstrap_balance_ratio is not None else np.nan),
            "criterion": np.str_(self.criterion),
            "class_weights": class_weights,
            "min_samples_split": np.int32(self.min_samples_split),
            "min_samples_leaf": np.int32(self.min_samples_leaf),
            "is_fitted": np.bool_(self.is_fitted),}

    
    @classmethod
    @cpu_limit(0.9)
    @memory_limit(0.9)
    @typechecked
    def load_model(cls, 
                   directory: str = r"../artifacts/forest/", 
                   silent_load: bool = False
                   ) -> Self:
        """
        ## Usage
        Use this function to load a saved the RF model.
        
        :param directory: Path to desired directory which the model will be saved in.
        :param silent_load: Silents the "Successful Loading" message.
        """
        # --- Checking for Model Existence ---
        filepath = os.path.join(directory, "random_forest_model.npz")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model data found at {filepath}!")
        # --- Loading Model ---
        loaded_model = cls()
        with np.load(filepath) as data:
            loaded_model._unpack_metadata(data)
            loaded_model.tree_list = cls._rebuild_trees(data)
        # --- Showing Message ---
        if not silent_load:
            print(f"Model Successfully Loaded!")
        # --- Returning ---
        return loaded_model

    def _unpack_metadata(self, data) -> None:
        self.n_trees = int(data["n_trees"])
        self.n_features = int(data["n_features"])
        self.n_samples = int(data["n_samples"])
        self.tree_depth = int(data["tree_depth"])
        self.trees_node_min_purity = float(data["trees_node_min_purity"])
        ttt = float(data["train_time_taken"])
        self.train_time_taken = ttt if not np.isnan(ttt) else None
        bb_kind = data["bootstrap_balance_kind"].item()
        if bb_kind == "none":
            self.bootstrap_balance = None
        elif bb_kind == "auto":
            self.bootstrap_balance = "auto"
        else:
            self.bootstrap_balance = float(data["bootstrap_balance_value"])
        bbr = float(data["bootstrap_balance_ratio"])
        self.bootstrap_balance_ratio = bbr if not np.isnan(bbr) else None
        self.criterion = data["criterion"].item()
        cw = data["class_weights"]
        self.class_weights = None if np.isnan(cw[0]) else (float(cw[0]), float(cw[1]))
        self.min_samples_split = int(data["min_samples_split"])
        self.min_samples_leaf = int(data["min_samples_leaf"])
        self.is_fitted = bool(data["is_fitted"])

    @staticmethod
    def _rebuild_trees(data) -> list:
        is_leaf, is_numerical = data["node_is_leaf"], data["node_is_numerical"]
        criteria, node_score = data["node_criteria"], data["node_score"]
        column_number = data["node_column"]
        left_child, right_child = data["node_left"], data["node_right"]
        prediction, probability = data["node_prediction"], data["node_probability"]
        tree_offsets = data["tree_offsets"]
        def _build(idx: int) -> Node:
            node = Node()
            node.is_leaf = bool(is_leaf[idx])
            if node.is_leaf:
                node.prediction = float(prediction[idx])
                node.probability = float(probability[idx])
            else:
                node.is_numerical = bool(is_numerical[idx])
                node.criteria = float(criteria[idx])
                node.node_score = float(node_score[idx])
                node.column_number = int(column_number[idx])
                node.left_child = _build(int(left_child[idx]))
                node.right_child = _build(int(right_child[idx]))
            return node
        return [_build(int(tree_offsets[i])) for i in range(len(tree_offsets) - 1)]
    
#> ---------------------------------------------------------------------------------------   
