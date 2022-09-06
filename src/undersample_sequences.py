"""Undersample GISAID metadata, select sequences by clade"""
import random
from collections import namedtuple, Counter
from pathlib import Path
from tqdm import tqdm
import pandas as pd
tqdm.pandas()

from parameters import PARAMETERS
SEED = PARAMETERS["SEED"]; random.seed(SEED)
PATH_METADATA = PARAMETERS["PATH_METADATA"]
CLADES = PARAMETERS["CLADES"]
SAMPLES_PER_CLADE = PARAMETERS["SAMPLES_PER_CLADE"]

print(">> Undersample sequences <<")
# Load metadata
COLS = ["Virus name", "Collection date", "Submission date","Clade", "Host", "Is complete?"]
data = pd.read_csv(PATH_METADATA,sep="\t", usecols=COLS)

# Remove NaN in Clades and not-complete sequences
data.dropna(axis="rows",
            how="any",
            subset=COLS, 
            inplace=True,
            )
data.drop_duplicates(subset=COLS, inplace=True)

# Filter by Clades and Host
CLADES = tuple(clade for clade in CLADES)
data.query(f"`Clade` in {CLADES} and `Host`=='Human'", inplace=True)

## Randomly select a subset of sequences
# Generate id of sequences in fasta file: "Virus name|Accession ID|Collection date"
data["fasta_id"] = data.progress_apply(lambda row: "|".join([
                                        row["Virus name"],
                                        row["Collection date"],
                                        row["Submission date"]
                                        ]
                                    ), axis=1)

# subsample 
SampleClade = namedtuple("SampleClade", ["fasta_id","clade"])
list_fasta_selected = []
for clade in tqdm(CLADES):
    samples_clade = data.query(f"`Clade` == '{clade}'")["fasta_id"].tolist()
    random.shuffle(samples_clade)
    # select 'SAMPLES_PER_CLADE' samples for each clade, or all of them if available samples are less than required
    list_fasta_selected.extend([SampleClade(fasta_id, clade) for fasta_id in samples_clade[:SAMPLES_PER_CLADE]])

Path("data/train").mkdir(exist_ok=True, parents=True)
fasta_selected = pd.DataFrame(list_fasta_selected)
fasta_selected.to_csv("data/train/undersample_by_clade.csv")
pd.Series(Counter(fasta_selected["clade"])).to_csv("data/train/selected_by_clade.csv")
pd.Series(Counter(data["Clade"])).to_csv("data/train/available_by_clade.csv")