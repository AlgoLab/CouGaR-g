"""
Random Forest and SVM for classification of SarsCov2
"""
configfile: "parameters.yaml"

KFOLD=5

rule train_rf:
    input: 
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    params:
        kfold=KFOLD
    output: 
        f"data/rf-svm/metrics-rf.csv",
    script: 
        "src/cv_random_forest.py {params.kfold}"

rule train_svm:
    input: 
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    params:
        kfold=KFOLD
    output: 
        f"data/train-rf/preprocessing-rf.json"
    script: 
        "src/cv_random_forest.py {params.kfold}"

