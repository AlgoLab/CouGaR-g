"""
Most relevant kmers for each clade based on shap values
"""
from tqdm import tqdm
import pandas as pd
import numpy as np 
import shap 

from pathlib import Path
from loaders.model_loader import ModelLoader
from fcgr.utils import (
    fcgrpos2kmers,
    kmer_importance_shap
)
from preprocessing import Pipeline
from parameters import PARAMETERS

CLADES = PARAMETERS["CLADES"]
KMER = PARAMETERS["KMER"]
PREPROCESSING = [(k,v) for k,v in PARAMETERS["PREPROCESSING"].items()]
MODEL_NAME  = f"resnet50_{KMER}mers"

# get best weights
CHECKPOINTS  = [str(path) for path in Path("data/train/checkpoints").rglob("*.hdf5")]
epoch_from_chkp = lambda chkp: int(chkp.split("/")[-1].split("-")[1])
CHECKPOINTS.sort(key = epoch_from_chkp)
BEST_WEIGHTS =  CHECKPOINTS[-1]
print(f"using weights {BEST_WEIGHTS} for shap")

PATH_SMAP = Path(f"data/saliency_map") # to access (centroid) representative FCGR
PATH_SHAP = Path(f"data/shap_values")
for clade in CLADES: 
    PATH_SHAP.joinpath(f"{clade}").mkdir(exist_ok=True, parents=True)


# load model 
loader = ModelLoader()
model = loader(model_name = MODEL_NAME,
               n_outputs  = len(CLADES),
               weights_path = BEST_WEIGHTS
            )

# load preprocessing
preprocessing = Pipeline(PREPROCESSING)

# dict with position in FCGR to kmer
fcgrpos2kmer = fcgrpos2kmers(k=KMER) 

def f(x): 
    "Function inside explainer"
    tmp = x.copy()
    tmp = preprocessing(tmp)
    return model(tmp)

# define a masker that is used to mask out partitions of the input image.
masker = shap.maskers.Image("inpaint_telea", [2**KMER,2**KMER,1])

# create an explainer with model and image masker
explainer = shap.Explainer(f, masker, output_names=CLADES)

for clade in tqdm(CLADES, desc="Computing Shap Values"): 
    fcgr = np.load(PATH_SMAP.joinpath(f"{clade}/representative_FCGR.npy"))
    x = fcgr.copy()
    x = np.expand_dims(x, axis=-1) # channel axis
    x = np.expand_dims(x, axis=0) # batch axis
    shap_values = explainer(x, max_evals=1000, batch_size=8,)
    
    # shap values for clade
    idx_clade = CLADES.index(clade)
    sv = shap_values[0,:,:,0,idx_clade].values 
    THRESHOLD = max( abs(sv.min()), sv.max() ) / 2
    
    # extract most relevant kmers based on shap values 
    kmer_imp_sv = kmer_importance_shap(sv, threshold= THRESHOLD, fcgr = fcgr, fcgrpos2kmer=fcgrpos2kmer, Nmin = 50)
    df_kmer_imp = pd.DataFrame(kmer_imp_sv)
    
    # Save shap values and most relevant kmers
    path_save = PATH_SHAP.joinpath(f"{clade}")
    df_kmer_imp.to_csv(path_save.joinpath("relevant_kmers.csv"))
    np.save(path_save.joinpath("shap_values.npy"), sv)