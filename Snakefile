configfile: "parameters.yaml"

## 9. generate plots
rule plots:
    input:
        "data/train/training_log.csv",
        "data/test/predictions.csv" 
    output:
        expand("data/plots/confusion_matrix_{kmer}mer.pdf", kmer=config["KMER"]),
        expand("data/plots/accuracy_{kmer}mer.pdf", kmer=config["KMER"]),
        expand("data/plots/loss_{kmer}mer.pdf", kmer=config["KMER"]),
    script: 
        "plots.py"

## 8. clustering metrics
rule clustering_metrics:
    input:
        "data/test/predictions.csv", 
        "data/test/embeddings.npy"
    output: 
        "data/test/clustering_metrics.csv"
    run: 
        "clustering_metrics.py"

## 7. classification metrics
rule classification_metrics:
    input:
        "data/test/predictions.csv", 
        "data/test/embeddings.npy"
    output: 
        "data/test/metrics.csv",
        "data/test/accuracy.txt",
        "data/test/curve_pr.pdf"
    run: 
        "classification_metrics.py"

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
rule split_data:
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