import numpy as np
import joblib
import os
from ..utils import Timer
from ..tree import Node


#> ---------------------------------------------------------------------------------------   
class RandomForest:
    # --- Documentation ---
    """
    --- Random Forest Class ---
    Random Forest framework implemented from scratch by "Sepanta Metanat"

    First edit: "2026/02/25"
    Last edit: "2026/07/26"
    """
    
    def __init__(self):
        # --- Training Related ---
        self.tree_list = None #> a list or an array to store Tree Objects.
        self.n_trees = None #> How many trees shall be created from data.
        self.n_features = None #> How many columns did our x array had.
        self.n_samples = None #> How many samples have been provided for training.
        self.tree_depth = None #> Each Tree's depth.
        self.trees_node_min_purity = None #> Minimum leaf node purity.
        self.train_time_taken = None #> Training time taken.
        # --- Flags ---
        self.is_fitted = False #> Is trained or not.
        # --- Instances ---
        self.forest_Timer = Timer()
        self.tree_Timer = Timer()


    def __str__(self):
        return(f"""\n--- Random Forest Object ---
        Details:
        > Trained Status: {self.is_fitted}
        > Number of Trees: {self.n_trees}
        """)
        

    def fit(self, x_array, y_array, 
            n_trees=5, 
            trees_max_depth=5, 
            min_leaf_purity=0.95,
            timer = True, 
            detailed_timer = False,
            true_y_value=1, false_y_value=0):
        # --- Documentation ---
        """
        #> Usage and Information
        Use this function to train RF model.
            
        #> Parameters Documentation:
        1. x_array: training x_array.
        2. y_array: training y_array.
        3. n_trees: Number of Trees you desire to train in RF model.
        4. trees_max_depth: Each tree's maximum depth.
        5. min_leaf_purity: Node's minimum purity to turn leaf.
        6. timer: True/False Flag for timer. 
        7. detailed_timer: True/False Flag for detailed time usage by the model.
        8. true_y_value: True value label's on y_array. If you didn't turn your y_array to boolean, use this parameter to handle it.
        9. false_y_value: False value label's on y_array. If you didn't turn your y_array to boolean, use this parameter to handle it.
        """
        # --- Initelizing Related Tools ---
        rng = np.random.default_rng()
        # --- Validating Input ---
        x_array, y_array, timer, detailed_timer = self._input_validator(x_array, y_array,
                                                                        true_y_value, false_y_value,
                                                                        timer, detailed_timer)
        # --- Storing Information ---
        self._value_reset(x_array, n_trees, trees_max_depth, min_leaf_purity)
        # --- Printing Details ---
        self._forest_timer_and_info("start", timer, detailed_timer)
        # --- Iterating to create Trees ---
        for i in range(n_trees):
            # --- Printing Details ---
            self._tree_timer_and_info("start", i, detailed_timer)
            # --- Creating Bootstraped Data ---
            x_array_boot, y_array_boot = self._data_bootstrapper(x_array, y_array, rng)
            # --- Creating the Tree ---
            Tree = Node()
            Tree.fit(x_array_boot, y_array_boot, trees_max_depth, min_leaf_purity, rng)
            # --- Saving the Tree ---
            self.tree_list.append(Tree)
            # --- Printing Details ---
            self._tree_timer_and_info("stop", i, detailed_timer)
            self._tree_timer_and_info("print_info", i, detailed_timer)
        # --- Printing Details ---
        self._forest_timer_and_info("stop", timer, detailed_timer)
        self._forest_timer_and_info("print_info", timer, detailed_timer)
        # --- Chagning the Flag ---
        self.is_fitted=True

    def _input_validator(self, x_array, y_array, 
                         true_y_value, false_y_value,
                         timer, detailed_timer):
        # --- Attempt Conversion to Numpy.Array ---
        if not isinstance(x_array, np.ndarray):
            try:
                x_array = np.asarray(x_array)
            except Exception:
                raise TypeError(f'Could not convert input of type "{type(x_array)}" to "numpy.array".')
        if not isinstance(y_array, np.ndarray):
            try:
                y_array = np.asarray(y_array)
            except Exception:
                raise TypeError(f'Could not convert input of type "{type(y_array)}" to "numpy.array".')    
        # --- Checking the Inputs Logic ---
        rows, columns = np.shape(x_array)
        if (columns>rows):
            x_array = x_array.T
        y_array = y_array.ravel()
        # --- Fixing the Inputs type ---
        if false_y_value != 0 or true_y_value != 1:
            y_array = np.where(y_array == true_y_value, 1, 0)
        # --- Fixing Timer Values ---
        if detailed_timer:
            timer = True
        # --- return ---
        return x_array, y_array, timer, detailed_timer
    
    def _value_reset(self, x_array, n_trees, trees_max_depth, min_leaf_purity):
        self.n_samples, self.n_features = np.shape(x_array)[0], np.shape(x_array)[1]
        if self.n_features is None:
            self.n_features = 1
        self.n_trees = n_trees
        self.tree_depth = trees_max_depth
        self.trees_node_min_purity = min_leaf_purity
        self.tree_list = []

    def _forest_timer_and_info(self, status, timer, detailed_timer):
        # --- Starting Timer ---
        if status == "start":
            if timer:
                self.forest_Timer.start()
            if detailed_timer:
                print("-- Random Forest Model Training Live Info --")
                print("> Model Tuning Info:")
                print(f"Forest Model Will Consist of {self.n_trees} Tree Objects.")
                print(f"Tree Depth Set to: {self.tree_depth}")
                print(f"Model Is Teaining on {self.n_samples} Samples.")
                print(f"Minimum Purity for Each Leaf Node is set to: {self.trees_node_min_purity*100}%")
                print("\n> Live Update:")
        # --- Stopping Timer ---
        elif status == "stop":
            self.forest_Timer.stop()
        # --- Printing Info ---
        elif status == "print_info":
            self.train_time_taken = self.forest_Timer.elapsed_time()
            print("\n> Model Has Been Trained Successfully.")
            print(f"Total Time Taken: {self.train_time_taken}")
        
    def _tree_timer_and_info(self, status, iteration, detailed_timer):
        # --- Starting Timer ---
        if detailed_timer:
            if status == "start":
                self.tree_Timer.start()
            # --- Stopping Timer ---
            elif status == "stop":
                self.tree_Timer.stop()
            # --- Returning Info ---
            elif status == "print_info":
                print(f"{iteration+1}st Tree Has Been Trained. Time Taken: {self.tree_Timer.elapsed_time()}")

    def _data_bootstrapper(self, x_array, y_array, rng):
        # --- Getting the Shape of X Array ---
        n_rows, n_columns= np.shape(x_array)
        # --- Choosing Random Rows ---
        random_rows = rng.choice(n_rows, size=n_rows)
        # --- Applying the Random Rows Mask ---
        if n_columns!=1:
            x_array_boot=np.array(x_array[random_rows,:])
        else: 
            x_array_boot=np.array(x_array[random_rows])
        y_array_boot=np.array(y_array[random_rows])
        # --- Returning ---
        return x_array_boot, y_array_boot
    

    def predict(self, x_array, detailed=True, timer=True):
        # --- Documentation ---
        """
        #> Usage and Information
        Use this function to use the trained RF model and predict.

        #> Parameters Documentation:
        1. x_array: training x_array.
        2. detailed: Detailed (not rounded) y predictions.
        3. timer: Flag for starting the timer.
        """
        # --- Quick Validation of Input ---
        n_rows, n_columns = np.shape(x_array)
        if n_columns!=self.n_features:
            raise TypeError(f"Entered X_Array's features does not match the trained.")
        # --- If Input has 1 Row ---
        if n_rows==1:
            # --- If the output should be Detailed ---
            if detailed:
                return self._predict_row(x_array, timer)
            else:
                return np.round(self._predict_row(x_array, timer))
        # --- If Input has n Rows ---
        else:
            # --- If the output should be Detailed ---
            if detailed:
                return self._predict_n_rows(x_array, timer)
            else:
                return np.round(self._predict_n_rows(x_array, timer))

    def _predict_row(self, row, timer):
        # --- Timer ---
        if timer:
            self._predict_timer("start")
        # --- Defining temp Variable ---
        predicted_y_probabilities = 0
        # --- Iterating over Trees ---
        for Tree in self.tree_list:
            predicted_y_probabilities += Tree.predict(row)
        # --- Timer ---
        if timer:
            self._predict_timer("stop")
            self._predict_timer("print_info")
        # --- Calculating Prediction Probability and Returning---
        return (predicted_y_probabilities/self.n_trees if self.n_trees!=0 else 0)

    def _predict_n_rows(self, x_array, timer):
        # --- Timer ---
        if timer:
            self._predict_timer("start")
        # --- Defining Saving List ---
        predicted_y_probabilities_array=[]
        # --- Iterating Over Rows ---
        for row in x_array:
            # --- Defining temp Variable ---
            predicted_y_probabilities = 0
            # --- Iterating over Trees ---
            for Tree in self.tree_list:
                predicted_y_probabilities+=Tree.predict(row)
            # --- Calculating the Prediction and Prediction Probability ---
            predicted_y_probabilities_mean = (predicted_y_probabilities/self.n_trees if self.n_trees!=0 else 0)
            predicted_y_probabilities_array.append(predicted_y_probabilities_mean)
        # --- Timer ---
        if timer:
            self._predict_timer("stop")
            self._predict_timer("print_info")
        # --- Correcting Return Format ---
        predicted_y_probabilities_array=np.array(predicted_y_probabilities_array)
        return predicted_y_probabilities_array.ravel()

    def _predict_timer(self, status):
            # --- Starting Timer ---
            if status == "start":
                self.forest_Timer.start()
            # --- Stopping Timer ---
            if status == "stop":
                self.forest_Timer.stop()
            # --- Printing Info ---
            if status == "print_info":
                print("\nModel Has Finished Predicting Successfully.")
                print(f"Total Time Taken: {self.forest_Timer.elapsed_time()}")

    
    def save_model(self, directory=r"..//artifacts//random_forest//", silent_save=False):
        # --- Documentation ---
        """
        #> Usage and Information
        Use this function to save the RF model.

        #> Parameters Documentation:
        1. directory: Path to desired directory which the model will be saved in.
        2. silent_save: Silent's the "Successful Saving" message.
        """
        # --- Train Check ---
        if not self.is_fitted:
            raise ValueError("Error: Requested to save an untrained model!")
        # --- Creating Directory and Path ---
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, "random_forest_model.joblib")
        # joblib.dump efficiently handles large NumPy arrays within the object
        # --- Saving ---
        joblib.dump(self, filepath)
        # --- Printing info ---
        if not silent_save:
            print(f"Model Successfully Has Been Saved!")


    @classmethod
    def load_model(cls, directory=r"..//artifacts//random_forest//", silent_load=False):
        # --- Documentation ---
        """
        #> Usage and Information
        Use this function to load a saved the RF model.
        
        #> Parameters Documentation:
        1. directory: Path to desired directory which the model will be saved in.
        2. silent_load: Silent's the "Successful Loading" message.
        """
        # --- Checking for Model Existence ---
        filepath = os.path.join(directory, "random_forest_model.joblib")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Error: No model found at {filepath}!")
        # --- Loading Model ---
        loaded_model = joblib.load(filepath)
        # --- Showing Message ---
        if not silent_load:
            print(f"Model Successfully Loaded!")
        # --- Returning ---
        return loaded_model
#> ---------------------------------------------------------------------------------------   