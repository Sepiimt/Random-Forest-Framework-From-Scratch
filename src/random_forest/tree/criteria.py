import numpy as np
from collections.abc import Generator
from ..api.typing import SplitCriterion, Iterable, IterableTuple, CriteriaTuple

#> ---------------------------------------------------------------------------------------

#> Note on architecture: every criterion below is just a different formula applied to the
#> same four numbers per candidate split - (left_total, left_true, right_total, right_true).
#> Extracting those counts (the sort/cumsum scan for numerical columns, the crosstab for
#> categorical ones) is the expensive part and is criterion-agnostic, so it happens exactly
#> once per column regardless of which criterion is active. This is also what keeps the
#> criterion fully independent of, and combinable with, the "Balanced/stratified bootstrap
#> per tree" (A) - the criterion only ever looks at whatever sample it is handed.
def rf_criteria_chooser(X: Iterable, 
                        Y: Iterable, 
                        random_row_indices: Iterable, 
                        sort_order_array: Iterable, 
                        is_numerical_mask: Iterable,
                        column_position_map: Iterable, 
                        criterion: SplitCriterion, 
                        class_weights: Iterable, 
                        min_samples_leaf: int, 
                        rng: Generator
                        ) -> CriteriaTuple:
    n_columns = X.shape[1]
    # --- Determening the Amount of Columns Choosing --- 
    feature_select_count = int(np.sqrt(n_columns)) #> Note: Selecting how many random features we should pick
    # --- Initialize the Generator and Generating n Unique and Random Numbers ---
    #> "rng" will initilize in the tree for better performance.
    random_column_indices = rng.choice(n_columns, size=feature_select_count, replace=False)
    # --- "Better" Means Different Things to Different Criteria ---
    #> Note: Gini and Class-Weighted Gini are impurities (lower is better). Hellinger
    #> Distance is a divergence (higher is better). Tracking the direction here - rather
    #> than smuggling a sign-flip into the score itself - means a fitted node's stored
    #> score is always the criterion's true, human-readable value.
    higher_is_better = _CRITERIA_DIRECTION[criterion]
    best_split = (None, None, None, None) #> (column_number, criteria, is_numerical, score)
    best_score = -np.inf if higher_is_better else np.inf
    node_true_count = Y[random_row_indices].sum()
    # --- Determening the Chosed Features' Scores ---
    for random_column_number in random_column_indices:
        criteria, is_numerical, score = _column_split_score(X, Y,
                                                            random_row_indices,
                                                            sort_order_array,
                                                            random_column_number,
                                                            is_numerical_mask,
                                                            column_position_map,
                                                            criterion,
                                                            class_weights,
                                                            min_samples_leaf,
                                                            node_true_count)
        if criteria is None:
            continue
        # --- Saving the Results ---
        is_improvement = (score > best_score) if higher_is_better else (score < best_score)
        if is_improvement:
            best_score = score
            best_split = (random_column_number, criteria, is_numerical, score)
    # --- Returning the Best Column in Selected ---
    return best_split #> (column_number, criteria, is_numerical, score)

#> ---------------------------------------------------------------------------------------

def _column_split_score(X: Iterable, 
                        Y: Iterable, 
                        random_row_indices: Iterable, 
                        sort_order_array: Iterable, 
                        random_column_number: int,
                        is_numerical_mask: Iterable, 
                        column_position_map: Iterable, 
                        criterion: SplitCriterion, 
                        class_weights, 
                        min_samples_leaf: int,
                        node_true_count: int):
    # --- Dividing based on Explicit, Precomputed Mask (NOT re-inferred per split) ---
    #> Note: dtype cannot tell an encoded category from a real numerical value once
    #> both are float64 - so this is a lookup into a mask decided once at fit-time,
    #> not a per-node dtype guess. See RandomForest._build_numerical_mask.
    is_numerical = bool(is_numerical_mask[random_column_number])
    criteria, score = _split_score_processor(X, Y,
                                            is_numerical,
                                            random_row_indices,
                                            sort_order_array,
                                            random_column_number,
                                            column_position_map,
                                            criterion,
                                            class_weights,
                                            min_samples_leaf,
                                            node_true_count)
    # --- return ---
    return criteria, is_numerical, score

#> ---------------------------------------------------------------------------------------

def _split_score_processor(X: Iterable, 
                           Y: Iterable,
                           is_numerical: bool,
                           random_row_indices: Iterable,
                           sort_order_array: Iterable,
                           random_column_number: int,
                           column_position_map: Iterable,
                           criterion: SplitCriterion,
                           class_weights,
                           min_samples_leaf: int,
                           node_true_count: int):
    # --- Calculating Independent Values outside the loop ---
    node_total_len = len(random_row_indices)
    # --- Breaking the Process Rule ---
    if node_total_len <= 1:
        return (None, np.inf)
    # --- Calculating Every Candidate Split's Raw, Unweighted Counts ---
    if is_numerical:
        (criteria_list, left_leaf_total_len,
         left_leaf_true_count, right_leaf_total_len,
         right_leaf_true_count) = _numerical_split_counts(X, Y,
                                                          random_row_indices,
                                                          sort_order_array,
                                                          random_column_number,
                                                          column_position_map)
    else:
        (criteria_list, left_leaf_total_len,
         left_leaf_true_count, right_leaf_total_len,
         right_leaf_true_count) = _none_numerical_split_counts(X, Y,
                                                      random_row_indices,
                                                      random_column_number,
                                                      node_true_count)
    # --- Check for Valid Splits (`min_samples_leaf` Floor on Both Sides) ---
    #> Note: this generalizes the old ">0 samples on both sides" rule - the default
    #> `min_samples_leaf=1` reproduces that exact behaviour; larger values additionally
    #> forbid a split from carving off a sliver of fewer than `min_samples_leaf` samples,
    #> regardless of which criterion is scoring the candidates below.
    valid_mask = (left_leaf_total_len >= min_samples_leaf) & (right_leaf_total_len >= min_samples_leaf)
    if not np.any(valid_mask):
        return None, np.inf  # No valid split possible
    # --- Subset to valid splits ---
    criteria_list = criteria_list[valid_mask]
    left_leaf_total_len = left_leaf_total_len[valid_mask]
    left_leaf_true_count = left_leaf_true_count[valid_mask]
    right_leaf_total_len = right_leaf_total_len[valid_mask]
    right_leaf_true_count = right_leaf_true_count[valid_mask]
    # --- Scoring Every Valid Candidate Under the Chosen Criterion ---
    score_function = _CRITERIA_SCORERS[criterion]
    scores = score_function(left_leaf_total_len, left_leaf_true_count,
                             right_leaf_total_len, right_leaf_true_count,
                             node_total_len, node_true_count, class_weights)
    # --- Saving Best Computed Score and its Criteria ---
    #> Note: We calculate index once to save performance
    higher_is_better = _CRITERIA_DIRECTION[criterion]
    best_score_index = np.argmax(scores) if higher_is_better else np.argmin(scores)
    # --- return ---
    return criteria_list[best_score_index], scores[best_score_index]  #> (Criteria, Score)

#> ---------------------------------------------------------------------------------------

def _numerical_split_counts(X: Iterable, 
                            Y: Iterable, 
                            random_row_indices: Iterable, 
                            sort_order_array: Iterable, 
                            random_column_number: int, 
                            column_position_map: Iterable
                            ) -> IterableTuple:
    # --- 1. Extract and sort X and Y ONLY ONCE ---
    #> Fetch the specific sort order for this column. sort_order_array only holds
    #> columns that were numerical at fit-time (see RandomForest._build_numerical_mask),
    #> so we look up this column's position within that reduced array, not its
    #> position in the full X.
    col_sort_order = sort_order_array[:, column_position_map[random_column_number]]
    # Apply indexing once and save to local variables. 
    # This prevents redundant memory allocation in the steps below.
    x_sorted = X[random_row_indices, random_column_number][col_sort_order]
    y_sorted = Y[random_row_indices][col_sort_order]
    # --- 2. Cumulative counts ---
    node_total_len = len(random_row_indices)
    cum_true = np.cumsum(y_sorted).astype(np.int32)
    cum_total = np.arange(1, node_total_len + 1).astype(np.int32)
    # --- 3. Criteria (midpoints) & Masking ---
    # Much faster now because x_sorted is already cached in memory
    mask = x_sorted[:-1] != x_sorted[1:]
    # --- Check for Valid Split ---
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    # --- 4. Calculating the Amount of Values in each Section ---
    left_leaf_true_count = cum_true[:-1][mask]
    left_leaf_total_len = cum_total[:-1][mask]
    right_leaf_true_count = cum_true[-1] - left_leaf_true_count
    right_leaf_total_len = node_total_len - left_leaf_total_len
    # --- 5. Creating Criteria List ---
    # Using the cached variable again
    criteria_list = (x_sorted[:-1][mask] + x_sorted[1:][mask]) / 2.0
    # --- return ---
    return criteria_list, left_leaf_total_len, left_leaf_true_count, right_leaf_total_len, right_leaf_true_count

def _none_numerical_split_counts(X: Iterable, 
                                 Y: Iterable, 
                                 random_row_indices: Iterable, 
                                 random_column_number: int, 
                                 node_true_count: int
                                 ) -> IterableTuple:
    # --- Return Concept ---
    #> Note: Our concept for returning values will be a arrays of calculated details in the right and left leaf.
    # --- Calculating the Amount of False and True y ---
    node_total_len = len(random_row_indices)
    # --- Count per category ---
    #> Note: We proccess to know which criteria has which amounts of trues and false.
    x_values, xy_crosstab = _none_numerical_crosstab(X, Y, random_row_indices, random_column_number)
    # --- Left Leaf Process ---
    left_leaf_total_len = xy_crosstab[:,0]
    left_leaf_true_count = xy_crosstab[:,1]
    # --- Right Leaf Process ---
    right_leaf_total_len = node_total_len - left_leaf_total_len
    right_leaf_true_count = node_true_count - left_leaf_true_count
    # --- return ---
    return x_values, left_leaf_total_len, left_leaf_true_count, right_leaf_total_len, right_leaf_true_count

#> ---------------------------------------------------------------------------------------

def _none_numerical_crosstab(X: Iterable, 
                             Y: Iterable, 
                             random_row_indices: Iterable, 
                             random_column_number: int
                             ) -> IterableTuple:
    # --- Extracting the Globally Pre-Encoded Integer Categories ---
    encoded_x = X[random_row_indices, random_column_number].astype(np.int32)
    # --- O(N) Frequency Counting Without Implicit Sorting ---
    amount_of_each_x = np.bincount(encoded_x)
    true_y_per_x = np.bincount(encoded_x, weights=Y[random_row_indices])
    # --- Isolating Categories Actually Present in this Node's Sample ---
    active_categories = np.nonzero(amount_of_each_x)[0]
    # --- Stacking the Counts into the Expected [k x 2] Array ---
    xy_crosstab = np.column_stack((
        amount_of_each_x[active_categories],
        true_y_per_x[active_categories]
    )).astype(np.int32)
    # --- Return ---
    return active_categories, xy_crosstab

#> ---------------------------------------------------------------------------------------
# --- Criterion Scoring Functions ---
#> Note: Every scorer below has the exact same signature and is vectorized over an array
#> of candidate splits (one entry per threshold/category). `class_weights` is accepted by
#> all three for a uniform call site even though only `_score_class_weighted_gini` uses it.

def _score_gini(left_total, 
                left_true, 
                right_total, 
                right_true, 
                node_total, 
                node_true, 
                class_weights):
    #> Note: Plain, unweighted Gini impurity, weighted by each child's share of the node.
    #> Algebraically identical for binary Y whether written as `2p(1-p)` or `1-p^2-(1-p)^2` -
    #> both branches of the original code used one or the other; this is now the single
    #> shared implementation for both numerical and categorical columns.
    left_p = left_true / left_total
    right_p = right_true / right_total
    left_gini = 2 * left_p * (1 - left_p)
    right_gini = 2 * right_p * (1 - right_p)
    left_weight = left_total / node_total
    right_weight = right_total / node_total
    return (left_weight * left_gini) + (right_weight * right_gini)  #> lower is better

def _score_class_weighted_gini(left_total, 
                               left_true, 
                               right_total, 
                               right_true, 
                               node_total, 
                               node_true, 
                               class_weights):
    #> Note: `class_weights` = (weight_for_0, weight_for_1), resolved once in
    #> `RandomForest.fit()` from the *original, whole* training Y - deliberately not from
    #> the per-tree bootstrap sample, so this stays meaningful and independent of whatever
    #> "Balanced/stratified bootstrap per tree" (A) already did to that tree's sample. The
    #> two techniques can be combined, or either used alone.
    weight_0, weight_1 = class_weights
    left_false = left_total - left_true
    right_false = right_total - right_true
    left_true_w = left_true * weight_1
    left_false_w = left_false * weight_0
    left_total_w = left_true_w + left_false_w
    right_true_w = right_true * weight_1
    right_false_w = right_false * weight_0
    right_total_w = right_true_w + right_false_w
    node_total_w = left_total_w + right_total_w
    left_p = left_true_w / left_total_w
    right_p = right_true_w / right_total_w
    left_gini = 2 * left_p * (1 - left_p)
    right_gini = 2 * right_p * (1 - right_p)
    left_weight = left_total_w / node_total_w
    right_weight = right_total_w / node_total_w
    return (left_weight * left_gini) + (right_weight * right_gini)  #> lower is better

def _score_hellinger(left_total, 
                     left_true, 
                     right_total, 
                     right_true, 
                     node_total, 
                     node_true, 
                     class_weights):
    #> Note: Hellinger Distance Decision Trees (Cieslak & Chawla, 2008). Scores a split by
    #> how differently it routes the node's positives vs. its negatives - measured as
    #> class-conditional rates *within this node*, never weighted by node/child size - which
    #> is what makes it inherently insensitive to class skew, with no explicit class
    #> weighting needed. `node_true`/`node_total` are constant across every candidate for a
    #> given node, so `node_false` below is that node's negative count, not a per-split one.
    node_false = node_total - node_true
    true_positive_rate = left_true / node_true                    #> share of the node's positives sent left
    false_positive_rate = (left_total - left_true) / node_false   #> share of the node's negatives sent left
    return np.sqrt((np.sqrt(true_positive_rate) - np.sqrt(false_positive_rate)) ** 2 +
                   (np.sqrt(1 - true_positive_rate) - np.sqrt(1 - false_positive_rate)) ** 2)  #> higher is better

#> ---------------------------------------------------------------------------------------

# --- Criterion Registry ---
#> Note: Placed after the scorers themselves so the dict literals below can reference them
#> directly; both dicts are resolved by name (not by definition order) the first time a
#> node actually searches for a split, long after module import has finished, so ordering
#> here is purely for readability.
_CRITERIA_SCORERS = {
    "gini": _score_gini,
    "class_weighted_gini": _score_class_weighted_gini,
    "hellinger": _score_hellinger,
}
_CRITERIA_DIRECTION = {
    "gini": False,                 #> impurity: lower is better
    "class_weighted_gini": False,  #> impurity: lower is better
    "hellinger": True,             #> divergence: higher is better
}

#> ---------------------------------------------------------------------------------------
