# Credit-Card-Fraud-Detection-Model
This repository contains an Applied Artificial Intelligence academic project focused on predicting fraudulent credit card transactions using machine learning.
The goal of this project is to build a highly accurate classification model to distinguish between legitimate and fraudulent transactions.The dataset contains 1,000,000 rows and 8 features. Because fraudulent transactions are naturally rare compared to standard transactions, addressing **data imbalance** was a primary focus of this project.
## Technologies & Libraries Used:
-Pandas & NumPy (Data manipulation)
-Scikit-Learn (Machine Learning, Preprocessing, Metrics)
-Matplotlib (Visualization)

### Methodology & Data Preprocessing:
-Data Exploration: Checked for null values (none found) and analyzed class distribution.
-Handling Imbalance: The original dataset was extremely unbalanced. To prevent the model from overfitting to the majority class, "downsampling" was applied using the "resample" function to equalize the number of '0' (non-fraud) and '1' (fraud) outputs.
-Feature Scaling: Applied `StandardScaler` to numerical columns (`distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`).
-Binary Features: Kept binary categorical features (like `used_chip`, `online_order`) intact using the `remainder='passthrough'` method in the column transformer.

#### Modeling & Evaluation: 
The data was split into 70% training and 30% testing sets. Two Decision Tree Classifiers were trained to compare different criteria
Model 1 (Gini Criterion): Achieved **99.86%** test accuracy.
Model 2 (Entropy Criterion):** Achieved **99.98%** test accuracy.

##### Confusion Matrix Highlights
The model demonstrated exceptional performance with minimal errors on the test set.
True Positives (Fraud correctly identified): 26,192
True Negatives (Legitimate correctly identified): 26,244
False Positives (Mistakenly flagged as fraud): Only 4
False Negatives (Missed frauds): Only 2

How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib`
3. Place the dataset (`card_transdata_son.csv`) in the root directory.
4. Run the Python script: `python AAI_final.py`
