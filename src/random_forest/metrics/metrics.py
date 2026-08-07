import numpy as np

#> ---------------------------------------------------------------------------------------

def cross_entropy(predicted_y_prob, y_array, eps=1e-12):
    #> Note: Penalizes confident wrong predictions brutally & Rewards calibrated probabilities
    p = np.clip(predicted_y_prob, eps, 1 - eps)
    y = y_array
    # --- Return ---
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
#> ---------------------------------------------------------------------------------------

def confusion_matrix(predicted_y, y_array, binarize_threshold):
    #> Note: The raw reality of our model's decisions
    #> Note: Layout matches sklearn.metrics.confusion_matrix(y_true, y_pred) exactly - rows
    #> are the true class, columns are the predicted class: [[TN, FP], [FN, TP]]. This is NOT
    #> just a style choice: the previous [[TP, FP], [TN, FN]] layout held the same counts but
    #> in different cells, so any visual/numeric diff against sklearn's confusion_matrix()
    #> output looked "wrong" even when the underlying counts were identical.
    # --- Creating a Parallel Y Array ---
    y_pred = _binarize(predicted_y, binarize_threshold)
    y_true = y_array.astype(int)
    # --- Calculating TP,FP,TN,FN ---
    #> Note: Format: (True Positive, False Positive, True Negetive, False Negetive)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    # --- Return ---
    return np.array([[TN, FP],
                     [FN, TP]])

#> ---------------------------------------------------------------------------------------

def accuracy(predicted_y, y_array, binarize_threshold):
    #> Note: Fraction of correct predictions
    y_pred = _binarize(predicted_y, binarize_threshold)
    return np.mean(y_pred == y_array)

#> ---------------------------------------------------------------------------------------

def precision(predicted_y, y_array, binarize_threshold):
    #> Note: Of the predicted positives, how many were correct?
    y_pred = _binarize(predicted_y, binarize_threshold)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FP = np.sum((y_pred == 1) & (y_array == 0))
    # --- Return ---
    return TP / (TP + FP) if (TP + FP) != 0 else 0

#> ---------------------------------------------------------------------------------------

def recall(predicted_y, y_array, binarize_threshold):
    #> Note: Of the true positives, how many did we catch?
    y_pred = _binarize(predicted_y, binarize_threshold)
    # --- Process ---
    TP = np.sum((y_pred == 1) & (y_array == 1))
    FN = np.sum((y_pred == 0) & (y_array == 1))
    # --- Return ---
    return TP / (TP + FN) if (TP + FN) != 0 else 0

#> ---------------------------------------------------------------------------------------

def f1_score(predicted_y, y_array, binarize_threshold):
    #> Note: Harmonic mean of precision and recall
    prec = precision(predicted_y, y_array, binarize_threshold)
    reca = recall(predicted_y, y_array, binarize_threshold)
    # --- If ---
    if (prec + reca)!=0:
        f1 = 2 * (prec * reca) / (prec + reca)
    else: 
        f1=0
    # --- Return ---
    return f1

#> ---------------------------------------------------------------------------------------

def brier_score(predicted_y, y_array):
    #> Note: Mean squared error of probabilities vs truth
    return (1/len(y_array))*(((y_array-predicted_y)**2).sum())

#> ---------------------------------------------------------------------------------------

def _binarize(predicted_y, threshold: float):
    return (predicted_y >= threshold).astype(int)

#> ---------------------------------------------------------------------------------------
