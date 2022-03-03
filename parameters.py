"""
SET PARAMETERS FOR ALL STEPS
"""
# -- Define parameters
# General
KMER = 8
SPECIE = "hCoV-19"

# Undersample sequences
PATH_METADATA = "/data/GISAID/metadata.tsv"
CLADES = ['S','L','G','V','GR','GH','GV','GK','GRY']
SAMPLES_PER_CLADE = 5000
PATH_FASTA_GISAID = "/data/GISAID/sequences.fasta"

# For training
TRAIN_SIZE = 0.8 # size for val and test sets = (1-TRAIN_SIZE)/2
BATCH_SIZE = 16
EPOCHS = 50
MODEL = f"resnet50_{KMER}mers"
PREPROCESSING = [("divide_by_max",{})] # ("name_function", dict_kwargs)
WEIGHTS_PATH = None

# ---------------------
# Load to a Dictionary
PARAMETERS = dict(
    KMER = KMER,
    SPECIE = SPECIE,
    CLADES = CLADES,
    SAMPLES_PER_CLADE = SAMPLES_PER_CLADE,
    PATH_FASTA_GISAID = PATH_FASTA_GISAID,
    PATH_METADATA = PATH_METADATA, 
    FOLDER_FASTA = f"data/{SPECIE}",
    FOLDER_FCGR = f"fcgr-{KMER}-mer",
    TRAIN_SIZE = TRAIN_SIZE,
    BATCH_SIZE = BATCH_SIZE,
    EPOCHS = EPOCHS,   
    MODEL = MODEL,
    PREPROCESSING = PREPROCESSING,
    WEIGHTS_PATH = WEIGHTS_PATH,
    SEED = 42,    
)