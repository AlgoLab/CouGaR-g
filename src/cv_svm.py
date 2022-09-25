"""classification using SVM
Use repeated kfold 
"""
import sys
import json
from tqdm import tqdm 
import time 
import random
import numpy as np
import pandas as pd

from parameters import PARAMETERS
from collections import OrderedDict
from pathlib import Path

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, RepeatedKFold, cross_val_score, cross_val_predict

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score
)
from metrics.matthews_corrcoef import mcc

# params
KFOLD=int(sys.argv[-1])
KMER = PARAMETERS["KMER"] 
FOLDER_FCGR = Path(f"data/fcgr-{KMER}-mer")
LIST_FASTA   = list(FOLDER_FCGR.rglob("*npy"))

CLADES = PARAMETERS["CLADES"]
KMER = PARAMETERS["KMER"]
SEED = PARAMETERS["SEED"]

PATH_SAVE = Path("data/rf-svm")
PATH_SAVE.mkdir(exist_ok=True, parents=True)

# extract label from path
label_from_path = lambda path: str(path).split("/")[-2]

# input and output for SVM with relevant kmers and labels
def build_io_svm(paths_fcgr):
    # list to save frequencies of each kmer for all fcgr
    freqs_fcgr = [] 
    labels = []

    # build train data
    for path_fcgr in tqdm(paths_fcgr, desc="Extracting kmers from fcgr"):
        # extract label
        label = label_from_path(path_fcgr)
        labels.append(label)
        
        # load fcgr
        fcgr = np.load(path_fcgr)

        # extract frecuency of each list_relevant_kmers in the fcgr
        freqs_fcgr.append(fcgr.flatten())
        
    return pd.DataFrame(freqs_fcgr).numpy(), np.array(labels)

# def build_io_svm(paths_fcgr):
#     "Auxiliar function to generate random datasets"
#     N = 1000
#     return np.random.rand(N,KMER**2), np.array([random.choice(CLADES) for _ in range(N)]) 

# cross validation
X,y = build_io_svm(LIST_FASTA) # input-output
rkf = RepeatedKFold(n_splits=KFOLD, n_repeats=1, random_state=SEED) # k-fold sets

# classifier
ti = time.time()
clf = make_pipeline(StandardScaler(), SVC(probability=True)) # pipeline with preprocessing
tf = time.time() - ti
with open("data/rf-svm/time-svm.json","w") as fp: 
    json.dump(fp,
            {
                "cross_validation": tf,
                "avg_kfold": tf/KFOLD
            }
    )

probs = cross_val_predict(
    clf, X, y, cv=rkf, method="predict_proba"
)

## metrics
def compute_metrics(y_true, y_proba): 

    y_pred = [CLADES[y.argmax()] for y in y_proba]
    
    # accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Matthews coefficient
    cm = confusion_matrix(y_true, y_pred)
    m_coeff = mcc(cm)

    return {
        "acc": accuracy,
        "mcc": m_coeff
    }

# collect indexes used in each kfold
metrics = []
for train_index, test_index in rkf.split(X):
    y_true = y[test_index]
    y_proba = probs[test_index]

    metrics.append(compute_metrics(y_true, y_proba))

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("data/rf-svm/metrics-svm.tsv", sep="\t")