import numpy as np
import os

#> ---------------------------------------------------------------------------------------
def cross_entropy(predicted_y_prob, y_array, eps=1e-12):
    #> Note: Penalizes confident wrong predictions brutally & Rewards calibrated probabilities
    p = np.clip(predicted_y_prob, eps, 1 - eps)
    y = y_array
    # --- Return ---
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
#> ---------------------------------------------------------------------------------------
def confusion_matrix(predicted_y, y_array):
    #> Note: The raw reality of our model’s decisions
    # --- Creating a Parallel Y Array ---
    y_pred = np.round(predicted_y).astype(int)
    y_true = y_array.astype(int)
    # --- Calculating TP,FP,TN,FN ---
    #> Note: Format: (True Positive, False Positive, True Negetive, False Negetive)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    # --- Return ---
    return np.array([[TP, FP],
                     [TN, FN]])

#> ---------------------------------------------------------------------------------------
def accuracy(predicted_y, y_array):
    #> Note: Fraction of correct predictions
    y_pred = np.round(predicted_y)
    return np.mean(y_pred == y_array)

#> ---------------------------------------------------------------------------------------
def precision(predicted_y, y_array):
    #> Note: Of the predicted positives, how many were correct?
    y_pred = np.round(predicted_y)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FP = np.sum((y_pred == 1) & (y_array == 0))
    # --- Return ---
    return TP / (TP + FP) if (TP + FP) != 0 else 0

#> ---------------------------------------------------------------------------------------
def recall(predicted_y, y_array):
    #> Note: Of the true positives, how many did we catch?
    y_pred = np.round(predicted_y)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FN = np.sum((y_pred == 0) & (y_array == 1))
    # --- Return ---
    return TP / (TP + FN) if (TP + FN) != 0 else 0

#> ---------------------------------------------------------------------------------------
def f1_score(predicted_y, y_array):
    #> Note: Harmonic mean of precision and recall
    prec = precision(predicted_y, y_array)
    reca = recall(predicted_y, y_array)
    # --- If ---
    if (prec + reca)!=0:
        f1 = 2 * (prec * reca) / (prec + reca)
    else: 
        f1=0
    # --- Return ---
    return f1

#> ---------------------------------------------------------------------------------------
def brier_Score(predicted_y, y_array):
    #> Note: Mean squared error of probabilities vs truth
    return (1/len(y_array))*(((y_array-predicted_y)**2).sum())

#> ---------------------------------------------------------------------------------------
def model_evaluator(predicted_y, y_test,
                    RFModel=None,
                    silent=False,
                    save=False, 
                    text_file_name="metrics.txt", 
                    save_path=r"..\\results\\benchmarks\\"):
    # --- Documentation ---
    """
    #> Usage and Information
    Use this function to evalate RF model and create a "metrics.txt" to save the evaluation info.
    
    #> Parameters Documentation:
    1. predicted_y: RF model predicted y. 
    2. y_test: y_test array.
    3. silent: Flag for printing the results.
    4. save: Flag for saving the results as ".txt" file.
    5. text_file_name: Name for saved ".txt" file.
    6. save_path: Save path for ".txt" file.
    """
    # --- Getting Information ---
    r1, r2, r3, = cross_entropy(predicted_y, y_test), confusion_matrix(predicted_y, y_test), accuracy(predicted_y, y_test)
    r4, r5, r6 = precision(predicted_y, y_test), recall(predicted_y, y_test), f1_score(predicted_y, y_test)
    r7 = brier_Score(predicted_y, y_test)
    # --- Writing Information ---
    r1=f"Cross_Entropy: {r1}"
    r2=f"Confusion Matrix: \n{r2}\n"
    r3=f"Accuracy: {r3}"
    r4=f"Precision: {r4}"
    r5=f"Recall: {r5}"
    r6=f"F1_Score: {r6}"
    r7=f"Brier_Score: {r7}"
    if RFModel != None:
        mr1, mr2, mr3, mr4, mr5 = RFModel.n_trees, RFModel.tree_depth , RFModel.trees_node_min_purity, RFModel.n_samples, RFModel.train_time_taken
        mr1=f"Total Tree Objects in the Forest: {mr1}"
        mr2=f"Depth of Each Tree Object: {mr2}"
        mr3=f"Minimum Purity of Each Leaf Node in Tree Objects: {mr3}"
        mr4=f"Model Training on {mr4} Samples."
        mr5=f"Total Time Taken for Training the Model: {mr5}"
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
        path = os.path.join(save_path, text_file_name)
        tree_details = [mr1, mr2, mr3, mr4, mr5]
        details = [r1,r2,r3,r4,r5,r6,r7]
        with open(path, "w") as f:
            f.write("--- Model Info ---\n")
            if RFModel != None:
                for item in tree_details:
                    f.write("> " + item + "\n")
            f.write("\n--- Model Metrics ---\n")
            for item in details:
                f.write("> " + item + "\n")

#> ---------------------------------------------------------------------------------------
