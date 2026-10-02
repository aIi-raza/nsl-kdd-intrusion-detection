"""
main.py

Runs the full NSL-KDD intrusion detection pipeline end to end:
1. Load and combine the dataset
2. Clean data and create the binary target
3. Select the paper's 25 features, encode, and normalize
4. Train/test split
5. Train Decision Tree and Random Forest, evaluate, compare to the paper
6. Train XGBoost (extra model not used by the paper), evaluate
7. Add engineered features, retrain XGBoost, evaluate
8. Print the final comparison table and conclusion

Update DATA_DIR below to point at wherever you placed KDDTrain+.txt and
KDDTest+.txt on your machine.
"""

import os  # used to safely build file paths (works on Windows and Mac/Linux)

# Importing only the specific functions we need from each file in src/
from src.data_loader import load_and_combine_data, assign_column_names
from src.preprocessing import (
    drop_unused_column_and_create_target, select_features,
    encode_categorical_features, normalize_features, split_data
)
from src.feature_engineering import add_engineered_features
from src.train_models import train_decision_tree, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, build_comparison_table, print_explanation
from src.visualize import plot_confusion_matrices, plot_comparison_chart

# ----------------------------------------------------------------
# Update this path to wherever your dataset files are stored locally
# ----------------------------------------------------------------
DATA_DIR = "data"
# os.path.join builds a correct file path like "data/KDDTrain+.txt"
# (safer than writing "data/KDDTrain+.txt" directly, since slash direction
# differs between Windows and Mac/Linux)
TRAIN_FILE = os.path.join(DATA_DIR, "KDDTrain+.txt")
TEST_FILE = os.path.join(DATA_DIR, "KDDTest+.txt")


def main():
    # --- 1. Load data ---
    # Load both files and combine into one dataset, then name the columns
    full_df = load_and_combine_data(TRAIN_FILE, TEST_FILE)
    full_df = assign_column_names(full_df)

    # --- 2. Clean + target ---
    # Remove the unused column and create our 0/1 normal-vs-attack label
    full_df = drop_unused_column_and_create_target(full_df)
    # Keep only the paper's 25 important features
    df_selected = select_features(full_df)

    # --- 3. Encode + normalize ---
    # Turn text columns into numbers, then scale everything to 0-1
    df_selected = encode_categorical_features(df_selected)
    X_scaled, y, scaler = normalize_features(df_selected)

    # --- 4. Split ---
    # 80% for training, 20% held back for testing
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # --- 5. Train + evaluate DT and RF ---
    # Train both models on the training data
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Check how well each one performs on the held-back test data
    dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Draw the confusion matrices and the comparison chart (vs. the paper)
    plot_confusion_matrices(y_test, dt_model.predict(X_test), rf_model.predict(X_test))
    plot_comparison_chart(dt_results, rf_results)

    # --- 6. Train + evaluate XGBoost (extra model) ---
    # This model was NOT used in the paper - our own addition
    xgb_model = train_xgboost(X_train, y_train)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # --- 7. Feature engineering + retrain XGBoost (bonus) ---
    # Add our 7 extra engineered features on top of the original 25
    df_fe = add_engineered_features(df_selected)
    # Normalize and split this new, bigger feature set the same way as before
    X_fe, y_fe, _ = normalize_features(df_fe)
    X_train_fe, X_test_fe, y_train_fe, y_test_fe = split_data(X_fe, y_fe)

    # Retrain XGBoost on this improved feature set and check if it helped
    xgb_fe_model = train_xgboost(X_train_fe, y_train_fe)
    xgb_fe_results = evaluate_model(xgb_fe_model, X_test_fe, y_test_fe, "XGBoost + Feature Engineering")

    # --- 8. Final comparison + conclusion ---
    # Gather every model's results together into one dictionary
    your_results = {
        "DT (Yours)": dt_results,
        "RF (Yours)": rf_results,
        "XGB (Yours)": xgb_results,
        "XGB_Tuned (Yours)": xgb_fe_results,
    }
    # Print the final table (ours vs. paper) and the written conclusion
    build_comparison_table(your_results)
    print_explanation()


# This is a standard Python convention: only run main() if this file is
# executed directly (e.g. "python main.py"), not if it gets imported elsewhere
if __name__ == "__main__":
    main()