# Classification of SARS-CoV-2 sequences using FCGR and CNN
Frequence Chaos Game Representation with Deep Learning

## Try the trained models
A [web app](https://huggingface.co/spaces/BIASLab/sars-cov-2-classification-fcgr) is available with all the trained models, you just need to upload a fasta file with your sequences

## Data
+ Sequences and metadata must be downloaded from [GISAID](https://www.gisaid.org/) after creating an account and accepting the Terms of Use. 
+ Reference sequence can be downloaded from [here](https://www.gisaid.org/resources/hcov-19-reference-sequence/).
+ List of variant markers for each clade are save in `mutations_reference.json` and can be found [here](https://www.gisaid.org/resources/statements-clarifications/clade-and-lineage-nomenclature-aids-in-genomic-epidemiology-of-active-hcov-19-viruses/)

**Before running the snakemake file, make sure to add them to `parameters.yaml`**
```yaml
PATH_FASTA_GISAID: "path/to/sequences.fasta"
PATH_METADATA: "path/to/metadata.tsv"
PATH_REFERENCE_GENOME: "path/to/reference.fasta"
```

## 
Create a virtual environment and install packages
```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Set parameters for the experiment in `parameters.yaml`
- See (and include) preprocessing functions at `preprocessing.py`

## Run

**Undersample sequences and generate FCGR** 
```bash
snakemake -s dataset.smk -p -c1
```

**Train 5 different neural networks. KFOLD will define the SEED for Repeated-KFold**
For this case KFOLD is an integer defining a different randomization of the datasets
for each KFOLD a different train, val and test sets will be generated, saved and used for training
```bash
snakemake -s crossval_nn.smk --config KFOLD=1 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=2 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=3 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=4 -p -c1
snakemake -s crossval_nn.smk --config KFOLD=5 -p -c1
```

**Train SVM and Random Forest with Cross Validation, 5-fold**
In this case, using sklearn, KFOLD is the number of folds we want to use, the random datasets are selected internally
```bash
snakemake -s rf_svm.smk --config KFOLD=5 -p -c1
```

to visualize a DAG with the rules
```bash
snakemake -s dataset.smk --forceall --dag | dot -Tpdf > dag_dataset.pdf
snakemake -s crossval_nn.smk --forceall --dag | dot -Tpdf > dag_crossval_nn.pdf
snakemake -s rf_svm.smk --forceall --dag | dot -Tpdf > dag_rf-svm.pdf
```
___
Snakefile runs codes in this order
1. `undersample_sequences.py`
2. `extract_sequences.py` (extract each undersample sequence in individuals fasta files)
3. `fasta2fcgr` (generates a npy file with the $k$th-FCGR for each extracted sequence in the previous step)
3. `split_data.py` (will create a file `datasets.json` with train, validation and test sets)
4. `train.py` (train the model for the $k$-mer selected)
5. `test.py`
6. `classification_metrics.py` (computes accuracy, precision, recall and f1-score) 
7. `clustering_metrics.py` (computes Silhouette score, Calinski-Harabaz and Generalized Discrimination Value in the test set)
8. `plots.py` (generates plot for accuracy and loss in the training and validation sets. Confusion matrix for the test set)
9. `saliency_map.py` and `shap_values.py` (feature importance methods)
10. `svm_experiment.py` (train a SVM using subsets of relevant kmers chosen by the feature importance methods)
11. `match_relevant_kmers.py` (match relevant kmers chose by the feature importance methods to the list of marker variants for each clade)

A folder `data/` will be created to save all intermediate results: 
```
data/
├── fcgr-6-mer
├── hCoV-19
├── matches
├── plots
├── saliency_map
├── shap_values
├── svm
├── test
├── train
└── rf-svm
```

<!-- - `<SPECIE>/` with all sequences extracted individually in the fasta file, in separated folders by label (Clade) 
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
- `rf-svm/` will contain metrics for SVM and RF
A folder `fcgr-<KMER>-mer/` will contain all the FCGR created from the sequences in `data/<SPECIE>`  -->
