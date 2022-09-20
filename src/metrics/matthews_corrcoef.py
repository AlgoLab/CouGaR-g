"""Matthews correlation coefficient for multiclass case
see https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef
"""

import numpy as np 

def mcc(confusion_matrix: np.array) -> float:
    "Matthews correlation coefficient for multiclass classification"
    # number of times each class 'k' truly occurred
    t = confusion_matrix.sum(axis=1)

    # number of times class 'k' was predicted
    p = confusion_matrix.sum(axis=0)

    # total number of samples correctly predicted 
    c = np.diag(confusion_matrix).sum()

    # total number of samples
    s = confusion_matrix.sum().sum()

    # matthews correlation coefficient
    return (c*s - (t*p).sum()) / np.sqrt((s**2-(p*p).sum())*(s**2-(t*t).sum()))