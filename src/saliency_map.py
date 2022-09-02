"""
Most relevant kmers for each clade based on saliency map
"""
import numpy as np
import pandas as pd 

from tqdm import tqdm
from pathlib import Path
from fcgr.fcgr import FCGR
from loaders.model_loader import ModelLoader
from feature_importance.saliency_maps import saliencymap
from fcgr.utils import (
    fcgrpos2kmers, 
    kmer_importance,
)
from preprocessing import Pipeline
from parameters import PARAMETERS

KMER = PARAMETERS["KMER"]
CLADES     = PARAMETERS["CLADES"] # order output model
PREPROCESSING = [(k,v) for k,v in PARAMETERS["PREPROCESSING"].items()]
MODEL_NAME  = f"resnet50_{KMER}mers"
PATH_SMAP = Path("data/saliency_map")
PATH_SMAP.mkdir(exist_ok=True)

# get best weights
CHECKPOINTS  = [str(path) for path in Path("data/train/checkpoints").rglob("*.hdf5")]
epoch_from_chkp = lambda chkp: int(chkp.split("/")[-1].split("-")[1])
CHECKPOINTS.sort(key = epoch_from_chkp)
BEST_WEIGHTS =  CHECKPOINTS[-1]
print(f"using weights {BEST_WEIGHTS} to test")

# load model 
loader = ModelLoader()
model = loader(model_name = MODEL_NAME,
               n_outputs  = len(CLADES),
               weights_path = BEST_WEIGHTS
            )

# load preprocessing
preprocessing = Pipeline(PREPROCESSING)

# fcgr
fcgr = FCGR(k=KMER)
fcgrpos2kmer = fcgrpos2kmers(k=KMER) # dict with position in FCGR to kmer

# Load predictions
predictions = pd.read_csv("data/test/predictions.csv")
predictions["TP"] = predictions.apply(lambda row: row["ground_truth"] == row["prediction"], axis=1)
# For each clade, compute the representative FCGR over all TP
for clade in tqdm(CLADES): 
    # path to save smap and relevant kmers for the clade
    PATH_CLADE = PATH_SMAP.joinpath(clade)
    PATH_CLADE.mkdir(exist_ok=True)

    # filter TP for each clade
    paths_tp_clade = predictions.query(f"`ground_truth`== '{clade}' and `TP` == True")["path"].tolist()

    # compute representative FCGR
    rep_fcgr = np.zeros((2**KMER,2**KMER))
    for path in paths_tp_clade: 
        fcgr = np.load(path)
        rep_fcgr = np.add(rep_fcgr, fcgr)
    rep_fcgr = rep_fcgr/len(paths_tp_clade) 

    # compute saliency map for the representative FCGR
    input_model = np.expand_dims(preprocessing(rep_fcgr), axis=0)
    smap, prob, pred_class = saliencymap(model, input_model, order_output=CLADES)

    # compute most relevant kmers
    list_kmers = kmer_importance(smap, 0.1, rep_fcgr, fcgrpos2kmer)
    if len(list_kmers)==0:
        list_kmers = kmer_importance(smap, 0, rep_fcgr, fcgrpos2kmer)
    
    np.save(file = PATH_CLADE.joinpath("saliency_map.npy"), arr = smap)
    np.save(file = PATH_CLADE.joinpath("representative_FCGR.npy"), arr = rep_fcgr)
    pd.DataFrame(list_kmers).to_csv(PATH_CLADE.joinpath("relevant_kmers.csv"))