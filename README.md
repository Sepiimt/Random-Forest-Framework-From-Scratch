
---

![Random_Forest_Thumbnail](https://i.postimg.cc/DyThbP9B/Gemini_Generated_Image_i0p16xi0p16xi0p1.png)


---
## **🔭 Project Overview**
This project features a robust, from-scratch implementation of the Random Forest algorithm using only NumPy. While high-level libraries like Scikit-Learn are industry standards, building the algorithm from the ground up allows for deep optimization of the decision-making process—specifically for high-stakes domains like Money Laundering Detection.

Money laundering datasets are notoriously imbalanced and complex. This implementation moves beyond simple nested loops and focuses on high-performance NumPy vectorization and flexible handling of both categorical and numerical financial data.

---
## 🚀 The "Secret Sauce": Technical Highlights

#### _⚡ Vectorized Gini Impurity_
Most "from scratch" implementations suffer from $O(n^2)$ complexity when finding splits. This project optimizes the search for the best criteria using:
* **Cumulative Sums (`np.cumsum`):** In `NumericalLeafGiniCalculator`, we sort the data once and use cumulative counts to evaluate every possible split point in $O(n \log n)$ time.
* **Crosstabs with `np.bincount`:** For categorical data, the `NoneNumericalCrosstab` function maps categories to integers and counts occurrences using fast C-level operations rather than Python loops.

#### _📊 Hybrid Feature Support_
Money laundering data is rarely just numbers. This model handles:
* **Numerical Data:** Automatic midpoint calculation for continuous variables (e.g., Transaction Amount).
* **Categorical Data:** Exact match splitting for discrete labels (e.g., Transaction Type, Flagged Country).

#### _🛡️ Anti-Overfitting Controls_
To handle the noise inherent in financial crime data, the model implements:
* **Feature Bagging:** Each tree only "sees" $\sqrt{n_{features}}$ random columns, preventing a single dominant feature from biasing the entire forest.
* **Purity-Based Pruning:** The `min_leaf_purity` parameter allows the model to stop splitting early if a node is "pure enough," preventing it from memorizing outliers.

---
## **🧹 Data Cleaning**
The initial dataset required preparation before analysis. This included handling missing values, correcting inconsistent entries, removing invalid or extreme outliers, and standardizing formats. These steps ensured the data was accurate, internally consistent, and suitable for model training.

#### 🗂️Portion of the raw data:

| `Time`   | `Date`     | Sender_account | `Receiver_account` | `Amount` | `Payment_currency` | `Received_currency` | `Sender_bank_location` | `Receiver_bank_location` | `Payment_type` | Is_laundering |
| -------- | ---------- | -------------- | ------------------ | -------- | ------------------ | ------------------- | ---------------------- | ------------------------ | -------------- | ------------- |
| 00:00:00 | 2024-01-01 | 75682867       | 78548252           | 20654.26 | UK pounds          | UK pounds           | Turkey                 | India                    | ACH            | 1             |
| 00:01:00 | 2024-01-01 | 66755036       | 60373415           | 56453.82 | US dollar          | US dollar           | UAE                    | Pakistan                 | Cross-border   | 1             |
| 00:02:00 | 2024-01-01 | 66882282       | 39679610           | 3301.28  | Indian rupee       | Indian rupee        | Morocco                | Italy                    | Cheque         | 0             |
| 00:03:00 | 2024-01-01 | 31081788       | 46161210           | 7848.25  | Euro               | Euro                | India                  | Austria                  | Cross-border   | 0             |
| 00:04:00 | 2024-01-01 | 23315092       | 63958064           | 52097.64 | Mexican peso       | Mexican peso        | Nigeria                | Mexico                   | Cross-border   | 1             |

#### 🛠️The Process:
The data is consistent and already processed to a acceptable extend; Though they're remains a few step necessary to take:
	
- **Converting Time Format:** For better distinguish, we transform the time format to hours only
	
- **Deleting `Sender_account` & `Receiver_account`:** Deleting the transaction ID's will help to remove the unnecessary features and prevent them to become the cornerstone of algorithm's decision making.

#### 🗂️Portion of the cleaned data:

| `Time` | `Amount` | `Payment_currency` | `Received_currency` | `Sender_bank_location` | `Receiver_bank_location` | `Payment_type` | Is_laundering |
| ------ | -------- | ------------------ | ------------------- | ---------------------- | ------------------------ | -------------- | ------------- |
| 0      | 20654.26 | UK pounds          | UK pounds           | Turkey                 | India                    | ACH            | 1             |
| 0      | 56453.82 | US dollar          | US dollar           | UAE                    | Pakistan                 | Cross-border   | 1             |
| 0      | 3301.28  | Indian rupee       | Indian rupee        | Morocco                | Italy                    | Cheque         | 0             |
| 0      | 7848.25  | Euro               | Euro                | India                  | Austria                  | Cross-border   | 0             |
| 0      | 52097.64 | Mexican peso       | Mexican peso        | Nigeria                | Mexico                   | Cross-border   | 1             |

---
## 📊 Visualization:
For better comprehension of data and feature relations: 

![Laundering_vs_Receiver_Bank_Location](https://i.postimg.cc/bwsH1W2v/Receiver_bank_location_vs_Is_laundering_plot.png)


![Laundering_vs_Sender_Bank_Location](https://i.postimg.cc/VkdBXpbL/Sender_bank_location_vs_Is_laundering_plot.png)


![Payment_type_Amount_plot](https://i.postimg.cc/CxzHbXfM/Payment_type_Amount_plot.png)

---
## 🧠 Core Concept: Why Random Forest?

A **Decision Tree** is a flowchart-like structure that splits data based on criteria that maximize "purity." While intuitive, a single tree is prone to **overfitting**—it essentially "memorizes" the training data, making it poor at detecting new, evolving money laundering tactics.

The **Random Forest** solves this by creating an ensemble of many independent trees. It introduces randomness in two ways:
	
1. **Bootstrap Aggregating (Bagging):** Each tree sees a different random subset of the data.
2. **Feature Bagging:** Each split only considers a random subset of features (e.g., $\sqrt{N}$).

**The result:** The forest trades the "perfect memory" of a single tree for the "generalized wisdom" of the crowd, significantly reducing False Positives.

---
## ⚙️ Logic & Process: A Deep Dive

### 1. The Decision to Refuse "Decoy" Data
In AML, datasets are notoriously **imbalanced** (e.g., 99.9% legitimate, 0.1% laundering). Many developers use "decoy data" (SMOTE or Oversampling) to balance classes. **I explicitly refused this approach.**

Generating synthetic data often introduces artificial noise and "hallucinated" patterns that do not exist in real financial crimes. By using the raw, imbalanced data, the model learns the **true signal** of the minority class. To make this work, the engine relies on the mathematical rigor of the Gini split rather than a balanced headcount.

### 2. Weighted Gini Impurity: The Mathematical Compass
The model searches for the best "Split Criteria" by minimizing the **Weighted Gini Impurity**. This is critical for imbalanced data. Instead of just looking for a majority, the weighted calculation evaluates how much "Information Gain" is achieved by a split relative to the size of the resulting branches.

The Gini Impurity $G$ for a node is:


$$G = 1 - \sum_{i=1}^{C} (P_i)^2$$

We then calculate the **Weighted Gini** for the split:


$$G_{weighted} = \frac{n_{left}}{n_{total}} G_{left} + \frac{n_{right}}{n_{total}} G_{right}$$

This ensures that even if a branch is small (like a rare laundering cluster), its "purity" is mathematically rewarded, allowing the model to isolate suspicious activity without needing a 50/50 class split.

### 3. High-Performance Vectorization
To handle large-scale financial logs, the implementation avoids Python `for-loops` for mathematical operations:
	
* **Numerical Optimization:** Uses `np.cumsum` and `np.argsort` to evaluate all possible numerical split points in a single pass ($O(n \log n)$).
* **Categorical Mapping:** Uses `np.bincount` and `np.unique` to handle non-numeric data (like Transaction Type or Currency) via fast C-level operations.

---
## 📂 Project Architecture

* **`forest.py`**: The Orchestrator. Handles bootstrapping, feature bagging, and probability aggregation (The `Forest` object).
* **`tree.py` (`Node` Class)**: The Recursive Engine. Grows the tree, handles leaf generation, and executes the `predict` logic.
* **`criteria.py`**: The Mathematician. Contains the vectorized logic for finding the optimal split across both numerical and categorical features.
* **`metrics.py`:** 
* **`utils.py`:** 

---
## 🚀 Usage

```python
from RandomForest import RandomForest

# Initialize and train
# Note: min_leaf_purity is set high to capture rare laundering signals
rf = RandomForest()
rf.fit(X_train, y_train, n_trees=25, trees_max_depth=10, min_leaf_purity=0.98)

# Get Risk Probabilities (Score from 0.0 to 1.0)
# In AML, we care about the probability of risk, not just a binary Yes/No.
risk_scores = rf.predict(X_test, detailed=True)

```

---
## 🛠 Key Technical Features

* **Zero Dependencies:** Pure NumPy and Python Standard Library.
* **Hybrid Data Support:** Native handling of categorical and continuous variables.
* **Probabilistic Output:** Provides a "Risk Score," allowing you to prioritize the most suspicious cases.

---
## **🖼️ Evaluation & Visualization**

Due to different model training, the cross-examined results is as follow:
![Result of Method One Scatter](https://i.postimg.cc/K8Rrtwgh/Config_vs_Result_bar_plot.png)

---
## 📈 Performance & Results

Dependent on your choice, the results can vary; Hence the best results were gathered and cross-examined for your better judgement.
You still can access the `.txt` log of the results at `results\metrics\`.

---
