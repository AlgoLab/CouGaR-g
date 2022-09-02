"""Divide FCGR generated into train, validation and test sets"""
import json
import pandas as pd
from parameters import PARAMETERS
from pathlib import Path
from data_selector import DataSelector

# Select all data
KMER = PARAMETERS["KMER"] 
FOLDER_FCGR = Path(f"data/fcgr-{KMER}-mer")
LIST_FASTA   = list(FOLDER_FCGR.rglob("*npy"))
TRAIN_SIZE   = float(PARAMETERS["TRAIN_SIZE"])
SEED = PARAMETERS["SEED"]

# Instantiate DataSelector
id_labels = [str(path) for path in LIST_FASTA] 
labels    = [path.parent.stem for path in LIST_FASTA] 
    
ds = DataSelector(
    id_labels,
    labels,
    SEED
    )

# Get train, test and val sets
ds(train_size=TRAIN_SIZE, balanced_on=labels)

with open("data/train/datasets.json", "w", encoding="utf-8") as f: 
    json.dump(ds.datasets["id_labels"], f, ensure_ascii=False, indent=4)

# Summary of data selected 
summary_labels =  pd.DataFrame(ds.get_summary_labels())
summary_labels["Total"] = summary_labels.sum(axis=1)
summary_labels.to_csv("data/train/summary_labels.csv")