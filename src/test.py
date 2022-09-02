import json 
import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import namedtuple
from pathlib import Path
from loaders.model_loader import ModelLoader
from loaders.data_generator import DataGenerator  
from preprocessing import Pipeline
from parameters import PARAMETERS

KMER = PARAMETERS["KMER"]
CLADES     = PARAMETERS["CLADES"] # order output model
PREPROCESSING = [(k,v) for k,v in PARAMETERS["PREPROCESSING"].items()]

MODEL_NAME  = f"resnet50_{KMER}mers"

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

preprocessing = Pipeline(PREPROCESSING)
            
# Option 2: from a list of files (test set)
with open("data/train/datasets.json", "r") as fp:
    dataset = json.load(fp)
list_test = dataset["test"]

print("Total files to test:", len(list_test))

# Instantiate DataGenerator for validation set
ds_test = DataGenerator(
    list_test,
    order_output_model = CLADES,
    batch_size = 1,
    shuffle = False,
    kmer = KMER,
    preprocessing = preprocessing,
) 


# Extract predictions and embeddings
list_preds = []
list_emb   = []

Preds = namedtuple("Preds",["path","ground_truth","prediction","confidence"])

for path, (input_model, labels_model) in tqdm(zip(list_test,iter(ds_test)),total=len(list_test)): 

    idx_gt = np.argmax(labels_model[0])
    gt = CLADES[idx_gt] # ground truth
    probs = model.predict(input_model)[0] # embedding last layer
    idx_pred = np.argmax(probs) # index with argmax
    pred = CLADES[idx_pred] # prediction of the model
    conf = probs[idx_pred] # confidence of the prediction

    # Collect results
    list_preds.append(
        Preds(path, gt, pred, conf)
    )

    list_emb.append(probs)

# Save results
path_save = Path("data/test")
path_save.mkdir(exist_ok=True)

pd.DataFrame(list_preds).to_csv(path_save.joinpath("predictions.csv"))
np.save(file = path_save.joinpath("embeddings.npy") ,arr = np.array(list_emb))