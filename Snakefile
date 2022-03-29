configfile: "parameters.yaml"

## 6. test model
rule test_model:
    input: 
        "data/train/datasets.json",
        "data/train/training_log.csv"
    output: 
        "data/test/embeddings.npy",
        "data/test/predictions.csv"
    script: "test.py"

## 5.  train model
rule train_model:
    input: 
        "data/train/datasets.json"
    output: 
        "data/train/training_log.csv",
        "data/train/preprocessing.json"
    script: 
        "train.py"

## 4. train, val, test sets
rule name:
    input:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"]), 
    output: 
        "data/train/datasets.json",
        "data/train/summary_labels.csv"
    script: 
        "split_data.py"    

## 3. Generate FCGR
rule generate_fcgr:
    input:
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    output:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    script:
        "fasta2fcgr.py"

## 2. Extract undersampled sequences in individual fasta files 
rule extract_sequences:
    input: 
        "data/train/undersample_by_clade.csv"
    params:
        specie = config["SPECIE"] 
    # output:
    #     "data/{params.specie}"
    output:
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    script: 
        "extract_sequences.py"


## 1. Undersample sequences from metadata
rule undersample_sequences:
    input: 
        config["PATH_METADATA"],
    output: 
        "data/train/undersample_by_clade.csv",
        "data/train/selected_by_clade.csv",
        "data/train/available_by_clade.csv",
    script: 
        "undersample_sequences.py"