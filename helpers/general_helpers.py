import os
import numpy as np 

def get_next_run_idx(folder, agent_name):
    """
    Finds the next available file index for saving outputs, given a folder and agent name

    Inputs:
    - folder (str): The folder where output files are stored or will be saved.
    - agent_name (str): The name used in the output file suffix (used to distinguish between agent outputs)

    Output: 
    (str): 3-digit string.
    """
    os.makedirs(folder, exist_ok=True)
    if agent_name == 'llm_explanation':
        existing = [fname for fname in os.listdir(folder) if f"_{agent_name}" in fname and fname.endswith(".txt")]
    else:
        existing = [fname for fname in os.listdir(folder) if fname.endswith(f"_{agent_name}.txt")]
        
    idxs = []
    for fname in existing:
        try:
            idx = int(fname[:3])
            idxs.append(idx)
        except ValueError:
            pass
    next_idx = max(idxs) + 1 if idxs else 1
    return f"{next_idx:03d}"


def get_interesting_idxs(y_test, y_pred):
    """
    Returns indices for the best, median and worst predictions

    Inputs:
    - y_test (array or pandas.Series): array-like or pandas Series containing true variable values (ground truth).
    - y_pred (array): Predicted target values (must be same length as y_test), it has the prediction of a Machine Learning Model

    Outputs: 
    interesting_idxs (list) : list of the indexes of interest.
    """
    y_true = np.asarray(y_test)
    preds = np.asarray(y_pred)

    errors = np.abs(y_true - preds)
    best_idx = errors.argmin()   
    worst_idx = errors.argmax()   

    median_price = np.median(y_true)
    median_idx = np.abs(y_true - median_price).argmin()

    expensive_idx = y_true.argmax()
    cheap_idx = y_true.argmin()

    interesting_idxs = [best_idx, worst_idx, median_idx, expensive_idx, cheap_idx]
    interesting_idxs = list(dict.fromkeys(interesting_idxs))
    return interesting_idxs

