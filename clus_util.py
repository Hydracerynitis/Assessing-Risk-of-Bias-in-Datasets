from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, accuracy_score
from clus.models.regression import RegressionEnsemble
import os
import subprocess
import shutil
import sys

def split_and_convert_to_clus_data(X, Y, test_indices):
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    X_test, Y_test = X.loc[test_indices], Y.loc[test_indices]
    X_train, Y_train = X.drop(index=test_indices), Y.drop(index=test_indices)
    
    labelled = Y_train.dropna().index
    X_train_lab, Y_train_lab = X_train.loc[labelled], Y_train.loc[labelled]
    X_train_unlab, Y_train_unlab = X_train.drop(index=labelled), Y_train.drop(index=labelled)

    # Set (partially) unlabeled data to completely unlabeled NaN
    Y_train_unlab = np.full_like(Y_train_unlab, np.nan, dtype=np.float32)

    np.savez("dataset_for_clus.npz", 
         X_train=X_train_lab, Y_train=Y_train_lab, 
         # X_train_unlab=X_train_unlab, Y_train_unlab=Y_train_unlab, 
         X_test=X_test, Y_test=Y_test)
    return X_train_lab, Y_train_lab, X_test, Y_test


def run_clus(X, Y, test_indices, seed=42, n_trees=3, tempdir="test", appendix="", debugmode=False):
    script_content = f"""
import sys
sys.path.append(".")
import clus_util
import numpy as np

def main():
    data = np.load("dataset_for_clus.npz")

    # Access variables
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]
    clus_util.run_clus_inner(X_train, Y_train, X_test, Y_test, 
        seed={seed}, n_trees={n_trees}, tempdir="{tempdir}", appendix="{appendix}")
    print("Execution finished.")

if __name__ == "__main__":
    main()
"""
    # split dataset and prepare for clus; write in a temporary file
    X_train, Y_train, X_test, Y_test = split_and_convert_to_clus_data(X, Y, test_indices)
    
    if os.path.exists('tempdirs'):
        shutil.rmtree('tempdirs')
    
    # Create a temporary file
    temp_script_path = "temp_run_clus.py"
    with open(temp_script_path, "w") as temp_script:
        temp_script.write(script_content)

    try:
        # Run the script
        result = subprocess.run([sys.executable, temp_script_path], capture_output=True, text=True)
        # result = subprocess.run(["python", temp_script_path], capture_output=True, text=True)
        if debugmode:
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)

        # Raise an error if the script failed
        result.check_returncode()
    except subprocess.CalledProcessError as e:
        print("Error while running the script:", e)
    finally:
        # Delete temporary scripts/folder after execution
        os.remove(temp_script_path)
        os.remove("dataset_for_clus.npz")
        if os.path.exists('tempdirs'):
            shutil.rmtree('tempdirs')

    true_vals = np.loadtxt(f'experiment_result/CLUS/CLUS_{n_trees}_truevals{appendix}.csv', delimiter=',', dtype=float)
    predictions = np.loadtxt(f'experiment_result/CLUS/CLUS_{n_trees}_predictions{appendix}.csv', delimiter=',', dtype=float)
    importances = pd.read_csv(f'experiment_result/CLUS/CLUS_{n_trees}_importances{appendix}.csv', delimiter=',', dtype=float)
    return true_vals, predictions, importances

def run_clus_inner(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 42,
    n_trees: int = 100,
    tempdir: str = "temp",
    appendix: str = ""
):
    """
    Run CLUS model and get predictions and probabilities.
    """
    # Multi-target or single-target?
    is_multi_target = (len(y_test[0]) > 1)
    
    model = RegressionEnsemble(
        verbose=0,
        is_multi_target=is_multi_target,
        random_state=seed,
        n_trees=n_trees,
        forest=[],
    )

    # Train model
    model, y_test_pred = model.fit(
        x_train, y_train, x_test, y_test, tempdir
    )
    # get the predictions from the output of the model
    true_vals = y_test_pred["true values"]
    pred_vals = y_test_pred[f"Forest with {n_trees} trees"]
    imp = model.feature_importances_[0][1]

    np.savetxt(f'experiment_result/CLUS/CLUS_{n_trees}_truevals{appendix}.csv', true_vals, delimiter=',', fmt='%s')
    np.savetxt(f'experiment_result/CLUS/CLUS_{n_trees}_predictions{appendix}.csv', pred_vals, delimiter=',', fmt='%s')
    pd.DataFrame(imp).to_csv(f'experiment_result/CLUS/CLUS_{n_trees}_importances{appendix}.csv', index=False)
