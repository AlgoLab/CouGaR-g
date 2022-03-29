# Classification of SARS-CoV-2 sequences using FCGR and CNN
Frequence Chaos Game Representation with Deep Learning

## Data
+ Sequences and metadata must be downloaded from [GISAID](https://www.gisaid.org/) after creating an account and accepting the Terms of Use. 
+ Reference sequence can be downloaded from [here](https://www.gisaid.org/resources/hcov-19-reference-sequence/).
+ List of variant markers for each clade are save in `mutations_reference.json` and can be found [here](https://www.gisaid.org/resources/statements-clarifications/clade-and-lineage-nomenclature-aids-in-genomic-epidemiology-of-active-hcov-19-viruses/)

## 
Create a virtual environment and install packages
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Set parameters for the experiment in `parameters.py`
- See (and include) preprocessing functions at `preprocessing.py`

Run codes in this order
1. `undersample_sequences.py`
2. `extract_sequences.py`
3. `fasta2fcgr`
3. `split_data.py` will create a file `datasets.json` with train, validation and test sets
4. `train.py`
5. `test.py`
6. `classification_metrics.py` 
7. `clustering_metrics.py` 
8. `plots.py`

A folder `data/` will be created to save all intermediate results: 
- `<SPECIE>/` with all sequences extracted individually in the fasta file, in separated folders by label (Clade) 
- `train/` will contain 
    - `undersample_by_clade.csv`
    - `available_by_clade.csv` a summary of the available sequences by clade, subject to the restrictions made in `undersample_sequences.py`(remove duplicates and empty rows)
    - `selected_by_clade.csv` a summary of the selected sequences by clade
    - `checkpoints/` will save the best weights during training.
    - `preprocessing.json` will save a list with the preprocessing applied to each FCGR during training.
    - `training_log.csv`: accuracy and loss and learning rate per epoch for train and validation sets.
    - `test/` will save all the metrics (classification and clustering) resulting from the evaluation of the best model on the test set.
    - `plots` accuracy and loss plots during training, confusion matrix
    - `saliency_maps/` representative FCGR by clade, saliency map and relevant k-mers for that representative.

A folder `fcgr-<KMER>-mer/` will contain all the FCGR created from the sequences in `data/<SPECIE>` 