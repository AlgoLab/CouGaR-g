configfile: "parameters.yaml"

## Extract undersampled sequences in individual fasta files 
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


## Undersample sequences from metadata
rule undersample_sequences:
    input: 
        config["PATH_METADATA"],
    output: 
        "data/train/undersample_by_clade.csv",
        "data/train/selected_by_clade.csv",
        "data/train/available_by_clade.csv",
    script: 
        "undersample_sequences.py"