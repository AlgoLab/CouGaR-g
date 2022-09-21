"""
Random Forest and SVM for classification of SarsCov2
"""
configfile: "parameters.yaml"

KFOLD=config["KFOLD"]

rule all:
    input:
        f"data/rf-svm/metrics-svm.tsv",
        f"data/rf-svm/metrics-rf.tsv"

rule train_rf:
    input: 
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    params:
        kfold=KFOLD
    output: 
        "data/rf-svm/metrics-rf.tsv",
    shell: 
        "python3 src/cv_random_forest.py {params.kfold}"

rule train_svm:
    input: 
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    params:
        kfold=KFOLD
    output: 
        "data/rf-svm/metrics-svm.tsv"
    shell: 
        "python3 src/cv_svm.py {params.kfold}"