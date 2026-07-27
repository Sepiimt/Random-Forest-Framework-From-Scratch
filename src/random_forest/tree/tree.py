import numpy as np
from .criteria import RandomForestCriteriaChooser

#> ---------------------------------------------------------------------------------------   
class Node:
    def __init__(self):
        # --- Training Related ---
        self.is_numerical = None #> Determines if the criteria value is numerical or not
        self.criteria = None #> Criteria Value
        self.node_gini = None #> Node's Criteria's Gini
        self.column_number = None #> the column's number which has been chosen.
        self.left_child = None #> Left child
        self.right_child = None #> Right child
        # --- Flags ---
        self.is_leaf = False #> If the node was leaf, then we have following arguments:
        # --- if Leaf Related ---
        self.prediction = None #> What is the prediction of the leaf.
        self.probability = None #> What is the probability of y=1.
        self.n_samples = None #> How many samples entered this leaf.
        self.n_features = None #> How many columns did our x array had.


    def __str__(self):
        str1=f"""\n --- Decision Tree Object ---\n
        Details:
        > Criteria: {self.criteria}
        > Criteria's Gini Impurity: {self.node_gini}"""
        if self.is_leaf==True:
            str2=f"""\n> Is Leaf: {self.is_leaf}
            > Leaf's Prediction: {self.prediction}
            > Leaf's Number of Samples: {self.n_samples}"""
            return str1+str2
        return str1


    def fit(self, x_array, y_array, max_depth, min_leaf_purity, rng):
        # --- Choosing Criteria ---
        #> Note: Return Format: (column_number, criteria, is_numerical, weighted_gini_impurity)
        gini_info = RandomForestCriteriaChooser(x_array, y_array, rng)
        # --- Catching Invalid Leaf Generating ---
        if gini_info[1] is None:
            self._make_leaf(y_array)
            return self
        # --- Saving Criteria and its Gini Information ---
        try:
            n_rows, n_columns = np.shape(x_array)
            self._info_regulator(gini_info, n_rows, n_columns)
        except ValueError:
            n_rows, n_columns = len(x_array), None
            self._info_regulator(gini_info, n_rows, n_columns)
        # --- Checking for Leaf Symptoms ---
        self._is_leaf_regulator(y_array, max_depth)
        if self.is_leaf is True:
            self._make_leaf(y_array)
            return self
        # --- Splitting Train Process Based on Criteria's Nature ---
        if self.is_numerical is True:
            self._numerical_fit(x_array, y_array, gini_info, max_depth, min_leaf_purity, rng)
        else:
            self._none_numerical_fit(x_array, y_array, gini_info, max_depth, min_leaf_purity, rng)

    def _numerical_fit(self, x_array, y_array, gini_info, max_depth, min_leaf_purity, rng):
        #> Note: Gini Details Format: (column_number, criteria, is_numerical, weighted_gini_impurity)
        if self.n_features is not None:
            left_child_values_mask = x_array[:,gini_info[0]]<=gini_info[1]
            right_child_values_mask = ~left_child_values_mask
        else:
            left_child_values_mask = x_array<=gini_info[1]
            right_child_values_mask = ~left_child_values_mask
        # -- Creating and Training Child Nodes ---
        self.left_child = Node()
        self.left_child.fit(x_array[left_child_values_mask], y_array[left_child_values_mask], max_depth-1, min_leaf_purity, rng)
        self.right_child = Node()
        self.right_child.fit(x_array[right_child_values_mask], y_array[right_child_values_mask], max_depth-1, min_leaf_purity, rng)
        
    def _none_numerical_fit(self, x_array, y_array, gini_info, max_depth, min_leaf_purity, rng):
        #> Note: Gini Details Format: column_number, criteria, is_numerical, weighted_gini_impurity)
        if self.n_features is not None:
            left_child_values_mask = x_array[:,gini_info[0]] == gini_info[1]
            right_child_values_mask = ~left_child_values_mask
        else:
            left_child_values_mask = x_array == gini_info[1]
            right_child_values_mask = ~left_child_values_mask
        # -- Creating and Training Child Nodes ---
        self.left_child = Node()
        self.left_child.fit(x_array[left_child_values_mask], y_array[left_child_values_mask], max_depth-1, min_leaf_purity, rng)
        self.right_child = Node()
        self.right_child.fit(x_array[right_child_values_mask], y_array[right_child_values_mask], max_depth-1, min_leaf_purity, rng)

    def _info_regulator(self, gini_info, n_samples, n_columns):
        #> Note: Format: (column_number, criteria, is_numerical, weighted_gini_impurity)
        self.column_number = gini_info[0]
        self.criteria = gini_info[1]
        self.is_numerical = gini_info[2]
        self.node_gini = gini_info[3]
        self.n_samples = n_samples
        self.n_features = n_columns
        
    def _is_leaf_regulator(self, y_array, max_depth, min_leaf_purity=0.95):
        # --- Depth Check ---
        if max_depth == 0:
            self.is_leaf = True
            return
        # --- Pure Node ---
        total = len(y_array)
        if total == 0:
            self.is_leaf = True
            return
        # --- Reached Minimum Purity ---
        positives = y_array.sum()
        p = positives / total
        if p >= min_leaf_purity or p <= (1 - min_leaf_purity):
            self.is_leaf = True
            return
    
    def _make_leaf(self, y_array):
        self.is_leaf = True
        total_len = len(y_array)
        total_true = y_array.sum()
        self.probability = total_true/total_len if total_len!=0 else 0
        self.prediction = np.round(self.probability)


    def predict(self, x_row):
        if self.is_leaf:
            return self.probability
        if self.is_numerical:
            if x_row[self.column_number] <= self.criteria:
                return self.left_child.predict(x_row)
            else:
                return self.right_child.predict(x_row)
        else:
            if x_row[self.column_number] == self.criteria:
                return self.left_child.predict(x_row)
            else:
                return self.right_child.predict(x_row)


#> ---------------------------------------------------------------------------------------   
    