import sys
import json 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

from collections import OrderedDict, namedtuple
from tqdm import tqdm
from pathlib import Path
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from fcgr.utils import fcgrpos2kmers
from parameters import PARAMETERS

feature_from = sys.argv[1] #"shap_values"

# base parameters
#minN, maxN = 1,15 # Number of relevant kmers to try
rangeN = [1,2,3,4,5,10,15,20,25,30,35,40,45,50]
CLADES = PARAMETERS["CLADES"]#['S','L','G','V','GR','GH','GV','GK','GRY']
KMER = PARAMETERS["KMER"]
kmer2pos = {kmer: pos for pos,kmer in fcgrpos2kmers(KMER).items()}

BASE_PATH = Path(f"data")

# load datasets
with open("data/train/datasets.json","r") as fp: 
    datasets = json.load(fp)
    
# add complete path to fcgr
train = datasets["train"]
test  = datasets["test"]


## FUNCTIONS
# extract label from path
label_from_path = lambda path: path.split("/")[-2]

# input and output for SVM with relevant kmers and labels
def build_io_svm(paths_fcgr, list_relevant_kmers, kmer2pos):
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
        freq_fcgr = OrderedDict()
        for kmer in list_relevant_kmers: 
            row,col=kmer2pos[kmer]
            freq_fcgr[kmer] = fcgr[row,col]
        freqs_fcgr.append(freq_fcgr)
        
    return pd.DataFrame(freqs_fcgr), labels

# extract N relevant kmers based on centroid results for shap-values or saliency-map
def extract_N_relevant_kmers_clade(clade, feature_from, N):
    path_relevant_kmers = BASE_PATH.joinpath(f"{feature_from}/{clade}/relevant_kmers.csv")
    df_relevant_kmers = pd.read_csv(path_relevant_kmers)
    N_relevant_kmers = df_relevant_kmers["kmer"].tolist()[:N]
    return N_relevant_kmers

# Collect relevant kmers
def collect_relevant_kmers_all_clades(feature_from, N, clades):
    relevant_kmers = set()
    for clade in clades: 
        relevant_kmers_clade = extract_N_relevant_kmers_clade(clade, feature_from, N)
        relevant_kmers = relevant_kmers.union(set(relevant_kmers_clade))

    list_relevant_kmers = list(relevant_kmers)
    list_relevant_kmers.sort() # sort lexicographically in ascending order
    
    return list_relevant_kmers

## EXPERIMENT

Results = namedtuple("Results", ["N","acc","total_kmers"])
results = []
for N in rangeN:
    print(f"Working on {N}")
    # collect relevant kmers for all clades
    list_relevant_kmers = collect_relevant_kmers_all_clades(feature_from, N, CLADES)
    total_kmers = len(list_relevant_kmers)
    
    # build input and output for train and test sets
    df_train, labels_train = build_io_svm(train, list_relevant_kmers, kmer2pos)
    df_test, labels_test = build_io_svm(test, list_relevant_kmers, kmer2pos)
    
    # train svm
    clf = make_pipeline(StandardScaler(), SVC())
    clf.fit(df_test, labels_test)

    # test svm
    predictions = clf.predict(df_test)
    acc = accuracy_score(y_true = labels_test, y_pred = predictions)
    
    # collect results
    results.append(Results(N,acc, total_kmers))

# Save results and plot 
path_save = BASE_PATH.joinpath(f"svm/{feature_from}")
path_save.mkdir(exist_ok=True, parents=True)
df_results = pd.DataFrame(results)
df_results.to_csv(path_save.joinpath(f"results_svm_{KMER}mer.csv"))
df_results.plot("N","acc")
plt.savefig(path_save.joinpath(f"N_vs_acc_svm_{KMER}mer.jpg"))