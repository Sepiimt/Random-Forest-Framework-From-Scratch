from .metrics import *
import os

#> ---------------------------------------------------------------------------------------

def model_evaluator(predicted_y: list | np.ndarray, 
                    y_test: list | np.ndarray,
                    binarize_threshold: float = 0.5,
                    RFModel: object | None = None,
                    silent: bool = False,
                    save: bool = False, 
                    text_file_name: str = "metrics.txt", 
                    save_path: str = r"../results/benchmarks/"):
    # --- Documentation ---
    """
    ## Function: Model Evaluator
    Use this function to evalate RF model and create a `metrics.txt` to save the evaluation info.
    
    :param predicted_y: RF model predicted y. 
    :param y_test: y_test array.
    :param silent: Flag for printing the results.
    :param save: Flag for saving the results as ".txt" file.
    :param text_file_name: Name for saved ".txt" file.
    :param save_path: Save path for ".txt" file.

    ## Author 
    - "Sepanta Metanat"
    """
    # --- Ensure 1D Array Shapes to Prevent Broadcasting Bugs ---
    predicted_y = np.asarray(predicted_y).ravel()
    y_test = np.asarray(y_test).ravel()
    # --- Getting Information ---
    r1, r2, r3, = cross_entropy(predicted_y, y_test), confusion_matrix(predicted_y, y_test, binarize_threshold), accuracy(predicted_y, y_test, binarize_threshold)
    r4, r5, r6 = precision(predicted_y, y_test, binarize_threshold), recall(predicted_y, y_test, binarize_threshold), f1_score(predicted_y, y_test, binarize_threshold)
    r7 = brier_score(predicted_y, y_test)
    # --- Writing Information ---
    r1=f"Cross_Entropy: {r1}"
    r2=f"Confusion Matrix: \n{r2}\n"
    r3=f"Accuracy: {r3}"
    r4=f"Precision: {r4}"
    r5=f"Recall: {r5}"
    r6=f"F1_Score: {r6}"
    r7=f"Brier_Score: {r7}"
    # --- Model Info (only exists if an RFModel was provided) ---
    tree_details = []
    if RFModel is not None:
        mr1, mr2, mr3, mr4, mr5 = RFModel.n_trees, RFModel.tree_depth , RFModel.trees_node_min_purity, RFModel.n_samples, RFModel.train_time_taken
        tree_details = [
            f"Total Tree Objects in the Forest: {mr1}",
            f"Depth of Each Tree Object: {mr2}",
            f"Minimum Purity of Each Leaf-Node in Tree Objects: {mr3}",
            f"Model Has Been Trained on {mr4} Samples.",
            f"Total Time Taken for Training the Model: {mr5}",
        ]
    # --- Repersenting Information ---
    if not silent:
        print(r1)
        print(r2)
        print(r3)
        print(r4)
        print(r5)
        print(r6)
        print(r7)
    # --- Saving Metrics ---
    if save:
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, text_file_name)
        details = [r1,r2,r3,r4,r5,r6,r7]
        with open(path, "w") as f:
            f.write("--- Model Info ---\n")
            for item in tree_details:
                f.write("> " + item + "\n")
            f.write("\n--- Model Metrics ---\n")
            for item in details:
                f.write("> " + item + "\n")

#> ---------------------------------------------------------------------------------------
