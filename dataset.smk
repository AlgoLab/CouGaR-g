# Extract sequences 
configfile: "parameters.yaml"


## 3. Generate FCGR
rule generate_fcgr:
    input:
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    output:
        expand("data/fcgr-{kmer}-mer/generated_fcgr.txt", kmer=config["KMER"])
    script:
        "src/fasta2fcgr.py"

## 2. Extract undersampled sequences in individual fasta files 
rule extract_sequences:
    input: 
        "data/train/undersample_by_clade.csv"
    output:
        expand("data/{specie}/extracted_sequences.txt", specie=config["SPECIE"])
    script: 
        "src/extract_sequences.py"


## 1. Undersample sequences from metadata
rule undersample_sequences:
    input: 
        config["PATH_METADATA"],
    output: 
        "data/train/undersample_by_clade.csv",
        "data/train/selected_by_clade.csv",
        "data/train/available_by_clade.csv",
    script: 
        "src/undersample_sequences.py"