import json
import os

base_dir = os.path.dirname(__file__)

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def make_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in text.split('\n')]
    }

def make_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in code.split('\n')]
    }

cells = []

# Title
cells.append(make_markdown_cell("# Bias Detection, Fairness Evaluation, and Mitigation\nThis notebook is the interactive version of the full research pipeline."))

# Installs
cells.append(make_markdown_cell("## 0. Install Dependencies"))
cells.append(make_code_cell("!pip install pandas numpy scikit-learn torch fairlearn shap scipy ucimlrepo"))

# Phase 1: Data
cells.append(make_markdown_cell("## 1. Data Fetching"))
cells.append(make_code_cell(read_file(os.path.join(base_dir, 'datasets', 'fetch_data.py'))))
cells.append(make_code_cell("df_adult = fetch_adult_income()\ndf_adult.head()"))

# Phase 2: Preprocessing
cells.append(make_markdown_cell("## 2. Preprocessing & Bias Symptoms"))
cells.append(make_code_cell(read_file(os.path.join(base_dir, 'src', '01_preprocessing_and_symptoms.py')).replace('if __name__ == "__main__":', 'if False:')))
cells.append(make_code_cell("df_processed = preprocess_data(df_adult, target_col='income', protected_attr='sex')\ncalculate_bias_symptoms(df_processed, target_col='income', protected_attr='sex')"))

# Phase 3: Baseline
cells.append(make_markdown_cell("## 3. Baseline Training"))
cells.append(make_code_cell(read_file(os.path.join(base_dir, 'src', '02_baseline_training.py')).replace('if __name__ == "__main__":', 'if False:')))
cells.append(make_code_cell("results, X_test, y_test = train_baseline(df_processed)"))

# Phase 4: Fairness Evaluation
cells.append(make_markdown_cell("## 4. Fairness Evaluation"))
cells.append(make_code_cell(read_file(os.path.join(base_dir, 'src', '03_fairness_evaluation.py')).replace('if __name__ == "__main__":', 'if False:')))
cells.append(make_code_cell("evaluate_model_fairness(results['RandomForest']['model'], X_test, y_test)\ny_pred = pd.Series(results['RandomForest']['model'].predict(X_test), index=X_test.index)\nmatched_counterpart_auditing(X_test, y_pred)"))

# Phase 5: Mitigation
cells.append(make_markdown_cell("## 5. Advanced Mitigation (Fair NN)"))
cells.append(make_code_cell(read_file(os.path.join(base_dir, 'src', '04_mitigation.py')).replace('if __name__ == "__main__":', 'if False:')))
cells.append(make_code_cell("X = df_processed.drop(columns=['income'])\ny = df_processed['income']\nprotected_attr_idx = X.columns.get_loc('sex')\nfair_model, _, _, _ = train_fair_model(X, y, protected_attr_idx)"))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = os.path.join(base_dir, 'Fairness_Pipeline.ipynb')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully created at: {out_path}")
