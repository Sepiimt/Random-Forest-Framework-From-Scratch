import numpy as np
from numba import njit
from collections.abc import Generator
from .criteria import rf_criteria_chooser
from ..api.typing import Iterable

#> ---------------------------------------------------------------------------------------

class Node:
    # --- Documentation ---
    """
    ## Class: Node
    Node class implemented from scratch by "Sepanta Metanat"
    - First edit: "2026/02/25"
    - Last edit: "2026/08/30"
    """
    # --- Slots ---
    __slots__ = (
        "is_numerical", "criteria", "node_score", 
        "column_number","left_child", "right_child", 
        "is_leaf", "prediction", "probability")


    def __init__(self):
        # --- Training Related ---
        self.is_numerical = None #> Determines if the criteria value is numerical or not
        self.criteria = None #> Criteria Value
        self.node_score = None #> Node's split score under the active criterion (lower=better for
                                #> "gini"/"class_weighted_gini", higher=better for "hellinger")
        self.column_number = None #> the column's number which has been chosen.
        self.left_child = None #> Left child
        self.right_child = None #> Right child
        # --- Flags ---
        self.is_leaf = False #> If the node was leaf, then we have following arguments:
        # --- if Leaf Related ---
        self.prediction = None #> What is the prediction of the leaf.
        self.probability = None #> What is the probability of y=1.


    def fit(self, 
            X: Iterable, 
            Y: Iterable, 
            random_row_indices: Iterable, 
            sort_order_array: Iterable, 
            max_depth: int, 
            min_leaf_purity: float, 
            is_numerical_mask: Iterable, 
            column_position_map: dict, 
            criterion: str, 
            class_weights, 
            min_samples_split: int, 
            min_samples_leaf: int,
            rng: Generator):
        # --- Checking for Leaf Symptoms ---
        self._is_leaf_regulator(Y, random_row_indices, max_depth, 
                                min_leaf_purity, min_samples_split)
        if self.is_leaf is True:
            self._make_leaf(Y, random_row_indices)
            return self
        # --- Choosing Criteria ---
        #> Note: Return Format: (column_number, criteria, is_numerical, score)
        gini_info = rf_criteria_chooser(X, Y, random_row_indices, sort_order_array, 
                                        is_numerical_mask, column_position_map,criterion, 
                                        class_weights, min_samples_leaf, rng)
        # --- Saving Criteria and its Score Information ---
        self._info_regulator(gini_info)
        # --- Catching Invalid Leaf Generating ---
        if gini_info[1] is None:
                    self._make_leaf(Y, random_row_indices)
                    return self
        # --- Splitting Train Process Based on Criteria's Nature ---
        if self.is_numerical is True:
            self._numerical_fit(X, Y, random_row_indices, sort_order_array, gini_info, 
                                max_depth, min_leaf_purity, is_numerical_mask, 
                                column_position_map, criterion, class_weights, 
                                min_samples_split, min_samples_leaf, rng)
        else:
            self._none_numerical_fit(X, Y, random_row_indices, sort_order_array, 
                                     gini_info, max_depth, min_leaf_purity, 
                                     is_numerical_mask, column_position_map, 
                                     criterion, class_weights, min_samples_split, 
                                     min_samples_leaf, rng)        

    def _numerical_fit(self, 
                       X: Iterable, 
                       Y: Iterable, 
                       random_row_indices: Iterable, 
                       sort_order_array: Iterable, 
                       gini_info: tuple, 
                       max_depth: int, 
                       min_leaf_purity: float, 
                       is_numerical_mask: Iterable, 
                       column_position_map: dict, 
                       criterion: str, 
                       class_weights, 
                       min_samples_split: int, 
                       min_samples_leaf: int, 
                       rng: Generator):
        #> Note: Gini Details Format: (column_number, criteria, is_numerical, score)
        (left_child_values_mask, 
                 right_child_values_mask) = self._left_child_mask_creator(
                     X, random_row_indices, gini_info, "numerical")
        # -- Masking Sort-Order Indices ---
        #> Note: sort_order_array only has columns for features that are numerical
        #> (see RandomForest._build_numerical_mask), so this remap is already cheaper
        #> whenever any categorical columns exist - nothing further needed here.
        left_sort_order = self._remap_sort_order(sort_order_array, left_child_values_mask)
        right_sort_order = self._remap_sort_order(sort_order_array, right_child_values_mask)
        # -- Creating and Training Child Nodes ---
        self.left_child = Node()
        self.left_child.fit(X, Y, 
                            random_row_indices[left_child_values_mask], 
                            left_sort_order, max_depth-1, min_leaf_purity, 
                            is_numerical_mask, column_position_map,
                            criterion, class_weights, min_samples_split, 
                            min_samples_leaf, rng)
        self.right_child = Node()
        self.right_child.fit(X, Y, 
                             random_row_indices[right_child_values_mask], 
                             right_sort_order, max_depth-1, min_leaf_purity, 
                             is_numerical_mask, column_position_map, criterion, 
                             class_weights, min_samples_split, min_samples_leaf, rng)
        
    def _none_numerical_fit(self, 
                            X: Iterable, 
                            Y: Iterable, 
                            random_row_indices: Iterable, 
                            sort_order_array: Iterable, 
                            gini_info: tuple, 
                            max_depth: int, 
                            min_leaf_purity: float, 
                            is_numerical_mask: Iterable, 
                            column_position_map: dict, 
                            criterion: str, 
                            class_weights, 
                            min_samples_split: int, 
                            min_samples_leaf: int, 
                            rng: Generator):
        #> Note: Gini Details Format: column_number, criteria, is_numerical, score)
        (left_child_values_mask, 
         right_child_values_mask) = self._left_child_mask_creator(
             X, random_row_indices, gini_info, "none_numerical")
        # -- Masking Sort-Order Indices ---
        left_sort_order = self._remap_sort_order(sort_order_array, left_child_values_mask)
        right_sort_order = self._remap_sort_order(sort_order_array, right_child_values_mask)
        # -- Creating and Training Child Nodes ---
        self.left_child = Node()
        self.left_child.fit(X, Y, random_row_indices[left_child_values_mask], 
                            left_sort_order, max_depth-1, min_leaf_purity, 
                            is_numerical_mask, column_position_map, criterion, 
                            class_weights, min_samples_split, min_samples_leaf, rng)
        self.right_child = Node()
        self.right_child.fit(X, Y, random_row_indices[right_child_values_mask], 
                             right_sort_order, max_depth-1, min_leaf_purity, 
                             is_numerical_mask, column_position_map, criterion, 
                             class_weights, min_samples_split, min_samples_leaf, rng)

    @staticmethod
    @njit
    def _left_child_mask_creator(X, random_row_indices, gini_info, status:str):
        if status == "numerical":
            #> Note: Gini Details Format: (column_number, criteria, is_numerical, score)
            left_child_values_mask = X[random_row_indices, gini_info[0]] <= gini_info[1]
            right_child_values_mask = ~left_child_values_mask
            # --- Return ---
            return left_child_values_mask, right_child_values_mask
        elif status == "none_numerical":
            #> Note: Gini Details Format: column_number, criteria, is_numerical, score)
            left_child_values_mask = X[random_row_indices, gini_info[0]] == gini_info[1]
            right_child_values_mask = ~left_child_values_mask
            # --- Return ---
            return left_child_values_mask, right_child_values_mask

    def _info_regulator(self, 
                        gini_info: tuple):
        #> Note: Format: (column_number, criteria, is_numerical, score)
        self.column_number = gini_info[0]
        self.criteria = gini_info[1]
        self.is_numerical = gini_info[2]
        self.node_score = gini_info[3]
        
    def _is_leaf_regulator(self, 
                           Y: Iterable, 
                           random_row_indices: Iterable, 
                           max_depth: int, 
                           min_leaf_purity: float, 
                           min_samples_split: int):
        # --- Depth Check ---
        if max_depth == 0:
            self.is_leaf = True
            return
        # --- Pure Node ---
        total = len(random_row_indices)
        if total == 0:
            self.is_leaf = True
            return
        # --- Structural Regularization: `min_samples_split` Floor ---
        #> Note: a node with fewer samples than this never even attempts a split search,
        #> regardless of its purity - independent of, and stackable with, the purity check
        #> below and the `min_samples_leaf` floor enforced inside the criteria chooser.
        if total < min_samples_split:
            self.is_leaf = True
            return
        # --- Reached Minimum Purity ---
        positives = Y[random_row_indices].sum()
        p = positives / total
        if p >= min_leaf_purity or p <= (1 - min_leaf_purity):
            self.is_leaf = True
            return
    
    def _make_leaf(self, Y: Iterable, 
                   random_row_indices: Iterable):
        self.is_leaf = True
        total_len = len(random_row_indices)
        total_true = Y[random_row_indices].sum()
        self.probability = total_true/total_len if total_len!=0 else 0
        self.prediction = np.round(self.probability)

    @staticmethod
    @njit
    def _remap_sort_order(sort_order_array: Iterable, 
                          mask: Iterable) -> np.ndarray:
        # No 2-D intermediates (child_mask, remapped) and no .copy() at all: walk each
        # column directly and write survivors straight into the output. Cheaper than
        # both the original vectorized form (which relied on NumPy's stride-based
        # boolean indexing - unsupported by Numba for >1-D masks) and the flatten+copy
        # workaround (which paid for two full N x n_features temporaries to fake it).
        N, n_features = sort_order_array.shape
        m = int(mask.sum())
        new_position_of_old = (np.cumsum(mask) - 1).astype(np.int32)  # parent pos -> child pos
        out = np.empty((m, n_features), dtype=sort_order_array.dtype)
        for col in range(n_features):
            write_row = 0
            for row in range(N):
                old_idx = sort_order_array[row, col]
                if mask[old_idx]:
                    out[write_row, col] = new_position_of_old[old_idx]
                    write_row += 1
        return out

    def predict(self, X: Iterable) -> np.ndarray:
        #> Pre-allocate output array corresponding to original row order
        predictions = np.empty(len(X), dtype=float)
        #> Start recursion with all row indices [0, 1, 2, ..., N-1]
        self._predict_recursive(X, np.arange(len(X)), predictions)
        # --- Return ---
        return predictions

    def _predict_recursive(self, X, indices, predictions):
        # --- Base Case: No Rows Routed to This Branch ---
        if len(indices) == 0:
            return
        # --- Base Case: Leaf Node ---
        if self.is_leaf:
            #> Assign leaf probability directly to original row positions
            predictions[indices] = self.probability
            return
        # --- Node Split Logic ---
        feature_vals = X[indices, self.column_number]
        if self.is_numerical:
            left_mask = feature_vals <= self.criteria
        else:
            left_mask = feature_vals == self.criteria
        #> Recurse down children, passing sub-indices
        self.left_child._predict_recursive(X, indices[left_mask], predictions)
        self.right_child._predict_recursive(X, indices[~left_mask], predictions)

    #> ---------------------------------------------------------------------------------------   
        
