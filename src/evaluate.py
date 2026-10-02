"""
evaluate.py

Calculates evaluation metrics for a trained model, builds the final
comparison table (your models vs. the paper's reported results), and
prints the explanation for why our best model did not beat the paper's
Random Forest.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)  # one function per metric needed to calculate

# The paper's reported results (Avci & Koca, 2023, Table 3)
# We hardcode these so we always have something fixed to compare our own results against
PAPER_RESULTS = {
    "DT (Paper)": {"accuracy": 99.51, "precision": 99.49, "recall": 99.46, "f1": 99.47},
    "RF (Paper)": {"accuracy": 99.72, "precision": 99.84, "recall": 99.56, "f1": 99.70},
}


def evaluate_model(model, X_test, y_test, model_name: str = "") -> dict:
    """
    Run predictions and calculate Accuracy, Precision, Recall, F1, and ROC-AUC
    for a trained model. Returns a dictionary of results (as percentages).
    """
    # Feed the test features into the trained model and get its guesses (0 or 1)
    preds = model.predict(X_test)

    # Compare the model's guesses (preds) against the real answers (y_test)
    # Each function returns a score from 0 to 1, so we multiply by 100 for a percentage
    results = {
        "accuracy": accuracy_score(y_test, preds) * 100,
        "precision": precision_score(y_test, preds) * 100,
        "recall": recall_score(y_test, preds) * 100,
        "f1": f1_score(y_test, preds) * 100,
        "roc_auc": roc_auc_score(y_test, preds) * 100,
    }

    # Only print results if a model_name was given (lets us skip printing if we don't want it)
    if model_name:
        print(f"\n  --- {model_name} Results ---")
        print(f"  Accuracy:  {results['accuracy']:.2f}%")
        print(f"  Precision: {results['precision']:.2f}%")
        print(f"  Recall:    {results['recall']:.2f}%")
        print(f"  F1-Score:  {results['f1']:.2f}%")
        print(f"  ROC-AUC:   {results['roc_auc']:.2f}%\n")

    return results


def build_comparison_table(your_results: dict) -> pd.DataFrame:
    """
    Build a single comparison table combining your model results with the
    paper's reported results.

    your_results should be a dict like:
        {
            "DT (Yours)": {...}, "RF (Yours)": {...},
            "XGB (Yours)": {...}, "XGB_Tuned (Yours)": {...}
        }
    """
    # {**dict1, **dict2} merges two dictionaries into one combined dictionary
    # This puts your results and the paper's results together in one place
    all_results = {**your_results, **PAPER_RESULTS}

    # Reshape the merged dictionary into a format pandas can turn into a table
    # list(...keys()) gets all model names, and each list comprehension pulls
    # out one metric from every model's results in the same order
    table_data = {
        "Model": list(all_results.keys()),
        "Accuracy (%)": [v["accuracy"] for v in all_results.values()],
        "Precision (%)": [v["precision"] for v in all_results.values()],
        "Recall (%)": [v["recall"] for v in all_results.values()],
        "F1-Score (%)": [v["f1"] for v in all_results.values()],
    }

    # Build the actual table and round every number to 2 decimal places for clean display
    comparison_table = pd.DataFrame(table_data).round(2)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON: YOURS vs. PAPER (Avci & Koca, 2023)")
    print("=" * 70)
    # to_string(index=False) prints the table without the extra row-number column
    print(comparison_table.to_string(index=False))
    print("=" * 70 + "\n")

    return comparison_table


def print_explanation():
    """Print the final conclusion explaining the results."""
    # A fixed block of text explaining our results and why XGBoost
    # didn't beat the paper's Random Forest score
    explanation = """
CONCLUSION:

Our Decision Tree and Random Forest results closely matched the paper's
reported numbers (within ~0.2%), confirming our implementation is valid
and consistent with published research on this dataset.

Our third model, XGBoost, outperformed our own DT and RF across nearly
every metric - demonstrating the value of a more advanced ensemble method.
Adding engineered features gave a further small improvement.

However, even our best model (XGBoost + Feature Engineering) did not
surpass the paper's reported Random Forest results. The most likely
reason is a preprocessing difference: the paper applied Cook's Distance
to detect and remove statistical outliers before training their models,
a step we did not replicate in this lab. Outlier removal can meaningfully
improve model performance on this dataset, and its absence here likely
explains the remaining gap.

This does not indicate a flawed implementation - it highlights that data
preprocessing quality can matter as much as model choice when working
with high-performing baselines like this one.
"""
    print(explanation)