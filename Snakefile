configfile: "parameters.yaml"

rule all:
    input:
        expand("data/matches/{feature_method}/{kmer}mers.csv", feature_method="shap_values", kmer=config["KMER"]),
        expand("data/matches/{feature_method}/{kmer}mers.csv", feature_method="saliency_map", kmer=config["KMER"]),
        expand("data/svm/{feature_method}/results_svm_{kmer}mer.csv", feature_method="shap_values", kmer=config["KMER"]),
        expand("data/svm/{feature_method}/results_svm_{kmer}mer.csv", feature_method="saliency_map", kmer=config["KMER"]),
        "data/test/clustering_metrics.csv",
        "data/test/metrics.csv",
        expand("data/plots/confusion_matrix_{kmer}mer.pdf", kmer=config["KMER"]),

## 12. match relevant kmers
# shap values
rule match_relevant_kmers_shap_values:
    input:
        expand("data/shap_values/{clade}/relevant_kmers.csv", clade=config["CLADES"]), 
        config["PATH_REFERENCE_GENOME"],
        "mutations_reference.json"
    params:
        feature_method="shap_values",
        relevant_kmers_to_match=50
    output: 
        expand("data/matches/{feature_method}/{kmer}mers.csv", feature_method="shap_values", kmer=config["KMER"])
    shell: 
        "python3 match_relevant_kmers.py {params.feature_method} {params.relevant_kmers_to_match}"

# saliency maps
rule match_relevant_kmers_saliency_map:
    input:
        expand("data/saliency_map/{clade}/relevant_kmers.csv", clade=config["CLADES"]), 
        config["PATH_REFERENCE_GENOME"],
        "mutations_reference.json"
    params:
        feature_method="saliency_map",
        relevant_kmers_to_match=50
    output: 
        expand("data/matches/{feature_method}/{kmer}mers.csv", feature_method="saliency_map", kmer=config["KMER"])
    shell: 
        "python3 match_relevant_kmers.py {params.feature_method} {params.relevant_kmers_to_match}"

## 11. svm experiment
# shap values
rule svm_shap_values:
    input:
        expand("data/shap_values/{clade}/relevant_kmers.csv", clade=config["CLADES"]), 
    params:
        feature_method="shap_values",
        kmer=config["KMER"]
    output: 
        expand("data/svm/{feature_method}/results_svm_{kmer}mer.csv", feature_method="shap_values", kmer=config["KMER"])
    shell: 
        "python3 svm_experiment.py {params.feature_method}"

# saliency map
rule svm_saliency_map:
    input:
        expand("data/saliency_map/{clade}/relevant_kmers.csv", clade=config["CLADES"]), 
    params:
        feature_method="saliency_map",
    output: 
        expand("data/svm/{feature_method}/results_svm_{kmer}mer.csv", feature_method="saliency_map", kmer=config["KMER"])
    shell: 
        "python3 svm_experiment.py {params.feature_method}"

## 10. feature importance methods
# shap values
rule shap:
    input:
        "data/test/predictions.csv",
        expand("data/saliency_map/{clade}/representative_FCGR.npy", clade=config["CLADES"]),
    output: 
        expand("data/shap_values/{clade}/relevant_kmers.csv", clade=config["CLADES"]),
        expand("data/shap_values/{clade}/shap_values.npy", clade=config["CLADES"]),
    script:
        "shap_values.py"

# saliency map
rule saliency_map:
    input:
        "data/test/predictions.csv"
    output: 
        expand("data/saliency_map/{clade}/relevant_kmers.csv", clade=config["CLADES"]),
        expand("data/saliency_map/{clade}/representative_FCGR.npy", clade=config["CLADES"]),
        expand("data/saliency_map/{clade}/saliency_map.npy", clade=config["CLADES"]),
    script:
        "saliency_map.py"

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
    script: 
        "clustering_metrics.py"

## 7. classification metrics
rule classification_metrics:
    input:
        "data/test/embeddings.npy",
        "data/test/predictions.csv", 
    output: 
        "data/test/metrics.csv",
        "data/test/accuracy.txt",
        "data/test/curve_pr.pdf"
    script: 
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
        "data/train/datasets.json",
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    output: 
        "data/train/training_log.csv",
        "data/train/preprocessing.json"
    script: 
        "train.py"

# ## 4. train, val, test sets
# rule split_data:
#     input:
#         expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"]), 
#     output: 
#         "data/train/datasets.json",
#         "data/train/summary_labels.csv"
#     script: 
#         "split_data.py"    

## 3. Generate FCGR
rule generate_fcgr:
    input:
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    output:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    script:
        "fasta2fcgr.py"

# ## 2. Extract undersampled sequences in individual fasta files 
# rule extract_sequences:
#     input: 
#         "data/train/undersample_by_clade.csv"
#     output:
#         expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
#     script: 
#         "extract_sequences.py"


# ## 1. Undersample sequences from metadata
# rule undersample_sequences:
#     input: 
#         config["PATH_METADATA"],
#     output: 
#         "data/train/undersample_by_clade.csv",
#         "data/train/selected_by_clade.csv",
#         "data/train/available_by_clade.csv",
#     script: 
#         "undersample_sequences.py"