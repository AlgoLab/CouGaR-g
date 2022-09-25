import sys
# Cross validation neural network training

configfile: "parameters.yaml"
KFOLD=config["KFOLD"] # be sure that the last correspond to k, the K-fold number, with k in [1,K]

rule all:
    input:
        f"data/test-{KFOLD}/clustering_metrics.csv",
        f"data/test-{KFOLD}/metrics.csv"

## 8. clustering metrics
rule clustering_metrics:
    input:
        f"data/test-{KFOLD}/predictions.csv", 
        f"data/test-{KFOLD}/embeddings.npy"
    params:
        kfold=KFOLD
    output: 
        f"data/test-{KFOLD}/clustering_metrics.csv"
    shell: 
        "python src/clustering_metrics.py {params.kfold}"

## 7. classification metrics
rule classification_metrics:
    input:
        f"data/test-{KFOLD}/embeddings.npy",
        f"data/test-{KFOLD}/predictions.csv", 
    params:
        kfold=KFOLD
    output: 
        f"data/test-{KFOLD}/metrics.csv",
        f"data/test-{KFOLD}/global_metrics.json",
        f"data/test-{KFOLD}/curve_pr.pdf"
    shell: 
        "python src/classification_metrics.py {params.kfold}"

## 6. test model
rule test_model:
    input: 
        f"data/train-{KFOLD}/datasets.json",
        f"data/train-{KFOLD}/training_log.csv"
    params:
        kfold=KFOLD
    output: 
        f"data/test-{KFOLD}/embeddings.npy",
        f"data/test-{KFOLD}/predictions.csv"
    shell: 
        "python src/test.py {params.kfold}"

## 5.  train model
rule train_model:
    input: 
        f"data/train-{KFOLD}/datasets.json",
        # expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    params:
        kfold=KFOLD
    output: 
        f"data/train-{KFOLD}/training_log.csv",
        f"data/train-{KFOLD}/preprocessing.json"
    shell: 
        "python src/train.py {params.kfold}"

## 4. train, val, test sets
rule split_data:
    input:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"]), 
    params:
        kfold=KFOLD
    output: 
        f"data/train-{KFOLD}/datasets.json",
        f"data/train-{KFOLD}/summary_labels.csv"
    shell: 
        "python src/split_data.py {params.kfold}"    
