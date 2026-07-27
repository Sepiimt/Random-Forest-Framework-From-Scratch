import numpy as np

#> ---------------------------------------------------------------------------------------
def RandomForestCriteriaChooser(x_array, y_array, rng):
    # --- Catching Error if X = 1D ---
    # --- If nD ---
    try:
        row_count, columns_count = np.shape(x_array)
        # --- Determening the Amount of Columns Choosing --- 
        feature_select_count = int(np.sqrt(columns_count)) #> Note: Selecting how many random features we should pick
        # --- Initialize the Generator and Generating n Unique and Random Numbers ---
        #> "rng" will initilize in the tree for better performance.
        random_ints = rng.choice(columns_count, size=feature_select_count, replace=False)
        # --- Saving our Random Feature's Ginis ---
        #> Note: Our save Format Will be: (column_number, criteria, is_numerical, weighted_gini_impurity)
        best_gini = (None, None, None, np.inf)
        # --- Determening the Chosed Features Gini Impurities ---
        for generated_random_column_number in random_ints:
            criteria, is_numerical, weighted_gini_impurity = GiniCalculator(x_array[:,generated_random_column_number], y_array)
            # --- Saving the Results ---
            if weighted_gini_impurity < best_gini[3] and criteria != None:
                best_gini = (generated_random_column_number, criteria, is_numerical, weighted_gini_impurity)
        # --- Returning the Best Column in Selected ---
        return best_gini #> (column_number, criteria, is_numerical, weighted_gini_impurity)
    # --- If 1D ---
    except ValueError:
        row_count, columns_count = len(x_array),0
        random_ints = 0
        # --- Saving our Random Feature's Ginis ---
        #> Note: Our save Format Will be: (column_number, criteria, is_numerical, weighted_gini_impurity)
        best_gini = (None, None, None, np.inf)
        # --- Determening the Chosed Features Gini Impurities ---
        criteria, is_numerical, weighted_gini_impurity=GiniCalculator(x_array, y_array)
        # --- Saving the Results ---
        best_gini = (0, criteria, is_numerical, weighted_gini_impurity)
        # --- Returning the Best Column in Selected ---
        return best_gini #> (column_number, criteria, is_numerical, weighted_gini_impurity)

#> ---------------------------------------------------------------------------------------
def GiniCalculator(x_array, y_array):
    # --- Dividing based on Input Type ---
    is_numerical = IsNumerical(x_array)
    criteria, weighted_gini_impurity=_GiniCalculatorProcessor(x_array, y_array, is_numerical)
    # --- return ---
    return criteria, is_numerical, weighted_gini_impurity

def IsNumerical(x_array): #> Note: To see if our X_Array is numerical or not.
    return np.issubdtype(x_array.dtype, np.number)

#> ---------------------------------------------------------------------------------------
def _GiniCalculatorProcessor(x_array, y_array, is_numerical):
    # --- Calculating Independent Values outside the loop ---
    node_total_len = len(x_array)
    # --- Breaking the Process Rule ---
    if node_total_len <= 1:
        return (None, np.inf)
    # --- Calculating Leaf's Gini ---
    if is_numerical:
        criteria_list, left_leaf_total_len, left_leaf_gini_impurity, right_leaf_total_len, right_leaf_gini_impurity = _NumericalLeafGiniCalculator(x_array, y_array)
    else:
        criteria_list, left_leaf_total_len, left_leaf_gini_impurity, right_leaf_total_len, right_leaf_gini_impurity = _NoneNumericalLeafGiniCalculator(x_array, y_array)
    # --- Check for valid splits (both sides must have >0 samples) ---
    valid_mask = (left_leaf_total_len > 0) & (right_leaf_total_len > 0)
    if not np.any(valid_mask):
        return None, np.inf  # No valid split possible
    # --- Subset to valid splits ---
    criteria_list = criteria_list[valid_mask]
    left_leaf_total_len = left_leaf_total_len[valid_mask]
    left_leaf_gini_impurity = left_leaf_gini_impurity[valid_mask]
    right_leaf_total_len = right_leaf_total_len[valid_mask]
    right_leaf_gini_impurity = right_leaf_gini_impurity[valid_mask]
    # --- Calculating the Weighteds ---
    left_leaf_weight = left_leaf_total_len / node_total_len
    right_leaf_weight = right_leaf_total_len / node_total_len
    # --- Calculating the Weighted Gini of the criteria ---
    #> Note: Summation of weighted Left and Right Gini Impurity
    weighted_gini_impurity = (left_leaf_weight * left_leaf_gini_impurity) + (right_leaf_weight * right_leaf_gini_impurity)
    # --- Saving Best Computed Weighted Gini and it's Criteria ---
    best_weighted_ginis_impurity_index = np.argmin(weighted_gini_impurity)  #> Note: We calculate index once to save performance
    return_weighted_ginis_impurity_details = (criteria_list[best_weighted_ginis_impurity_index], weighted_gini_impurity[best_weighted_ginis_impurity_index])
    # --- return ---
    return return_weighted_ginis_impurity_details  #> (Criteria, Weighted Gini)
    
def _NumericalLeafGiniCalculator(x_array, y_array):
    # --- Return Concept ---
    #> Note: Our concept for returning values will be a arrays of calculated details in the right and left leaf.
    # --- Sort x and align y ---
    #> Note: It is very important to sort our data, as we will be calculating criteria at the end and we need to calculate ginis the way we could know which gini belongs 
    #> to which criteria. We do it here once to preserve performance.
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = y_array[order]
    # --- Cumulative counts ---
    #> Note: "np.cumsum" will tell the sum of y values till our current index; if we consider y=[0,0,1,1,0,1], np.cumsum(y_array) will be [0,0,1,2,2,3]
    node_total_len = len(x_array)
    cum_true = np.cumsum(y_array)
    #> Note: we do "np.arange" to determine how many x we will have when we start iterating "np.cumsum(y_array)"
    cum_total = np.arange(1, node_total_len + 1)
    # --- Calculating the Amount of Values in each Section ---
    #> Note: How many true values will left leaf will have by every x added into it
    left_leaf_true_value_count=cum_true[:-1]
    left_leaf_total_len=cum_total[:-1]
    #> Note: How many true values will right leaf will have by every x added into it (we basically do the symmetry of the left leaf calculation)
    right_leaf_true_value_count=cum_true[-1] - left_leaf_true_value_count
    right_leaf_total_len=node_total_len - left_leaf_total_len
    # --- Probabilities ---
    left_leaf_p_of_true = np.zeros_like(left_leaf_true_value_count, dtype=float)
    np.divide(left_leaf_true_value_count, left_leaf_total_len, out=left_leaf_p_of_true, where=(left_leaf_total_len != 0))
    left_leaf_p_of_false = 1 - left_leaf_p_of_true
    right_leaf_p_of_true = np.zeros_like(right_leaf_true_value_count, dtype=float)
    np.divide(right_leaf_true_value_count, right_leaf_total_len, out=right_leaf_p_of_true, where=(right_leaf_total_len != 0))
    right_leaf_p_of_false = 1 - right_leaf_p_of_true
    # --- Gini impurities ---
    left_leaf_gini_impurity= (1 - left_leaf_p_of_true**2 - left_leaf_p_of_false**2)
    right_leaf_gini_impurity= (1 - right_leaf_p_of_true**2 - right_leaf_p_of_false**2)
    # --- Criteria (midpoints) ---
    mask = (x_array[:-1] != x_array[1:]) #> Note: We will create a mask to not compute the identical answers.
    criteria_list = ((x_array[:-1] + x_array[1:]) / 2)[mask] #> Note: We apply the mask.
    # --- Applying the Mask ---
    left_leaf_true_value_count = left_leaf_true_value_count[mask]
    left_leaf_total_len = left_leaf_total_len[mask]
    right_leaf_true_value_count = right_leaf_true_value_count[mask]
    right_leaf_total_len = right_leaf_total_len[mask]
    left_leaf_gini_impurity = left_leaf_gini_impurity[mask]
    right_leaf_gini_impurity = right_leaf_gini_impurity[mask]
    # --- return ---
    return criteria_list, left_leaf_total_len, left_leaf_gini_impurity, right_leaf_total_len, right_leaf_gini_impurity

def _NoneNumericalLeafGiniCalculator(x_array, y_array):
    # --- Return Concept ---
    #> Note: Our concept for returning values will be a arrays of calculated details in the right and left leaf.
    # --- Breaking the Process Rule ---
    # --- Calculating the Amount of False and True y ---
    node_total_len = len(x_array)
    total_true = y_array.sum()
    total_false = node_total_len - total_true
    # --- Count per category ---
    #> Note: We proccess to know which criteria has which amounts of trues and false.
    x_values, xy_crosstab = NoneNumericalCrosstab(x_array, y_array)
    # --- Left Leaf Process ---
    left_leaf_total_len = xy_crosstab[:,0]
    left_leaf_true_value_count = xy_crosstab[:,1]
    left_leaf_false_value_count = xy_crosstab[:,0] - xy_crosstab[:,1]
    # --- Right Leaf Process ---
    right_leaf_total_len = node_total_len - left_leaf_total_len
    right_leaf_true_value_count = total_true - left_leaf_true_value_count
    right_leaf_false_value_count = total_false - left_leaf_false_value_count
    # --- Probabilities ---
    left_leaf_p_of_true = np.zeros_like(left_leaf_true_value_count, dtype=float)
    np.divide(left_leaf_true_value_count, left_leaf_total_len, out=left_leaf_p_of_true, where=(left_leaf_total_len != 0)) 
    left_leaf_p_of_false = 1 - left_leaf_p_of_true
    right_leaf_p_of_true = np.zeros_like(right_leaf_true_value_count, dtype=float)
    np.divide(right_leaf_true_value_count, right_leaf_total_len, out=right_leaf_p_of_true, where=(right_leaf_total_len != 0))
    right_leaf_p_of_false = 1 - right_leaf_p_of_true
    # --- Gini impurities ---
    left_leaf_gini_impurity = 1 - left_leaf_p_of_true**2 - left_leaf_p_of_false**2
    right_leaf_gini_impurity = 1 - right_leaf_p_of_true**2 - right_leaf_p_of_false**2
    # --- return ---
    return x_values, left_leaf_total_len, left_leaf_gini_impurity, right_leaf_total_len, right_leaf_gini_impurity

#> ---------------------------------------------------------------------------------------
def NoneNumericalCrosstab(x_array, y_array):
    # --- Mapping Categories ---
    #> Note: First, we map unique strings to integers: O(n log n)
    #> then 'inv' is an array of indices representing the strings
    #> and 'names' are the original strings
    x_values, encoded_x = np.unique(x_array, return_inverse=True)
    # --- Counting True Ys per Category ---
    #> Note: Count 1s per category: O(n)
    #> y must be numeric (0s and 1s)
    true_y_per_x = np.bincount(encoded_x, weights=y_array, minlength=len(x_values)) #> Amount of True Values per Category.
    amount_of_each_x = np.bincount(encoded_x, minlength=len(x_values)) #> Amount of X Values per Category.
    xy_crosstab = np.column_stack((amount_of_each_x, true_y_per_x))
    # Stack them into a [k x 2] numpy array: O(k)
    return x_values, xy_crosstab
#> ---------------------------------------------------------------------------------------   