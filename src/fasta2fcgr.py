"""Create and FCGR.npy for each sequence.fasta in a folder"""
from pathlib import Path
from parameters import PARAMETERS
from fcgr.generate_fcgr import GenerateFCGR

SPECIE = PARAMETERS["SPECIE"]
FOLDER_FASTA = Path(f"data/{SPECIE}") 
LIST_FASTA   = list(FOLDER_FASTA.rglob("*fasta"))
KMER = PARAMETERS["KMER"] 
FOLDER_FCGR = Path(f"data/fcgr-{KMER}-mer")

#Instantiate class to generate FCGR
generate_fcgr = GenerateFCGR(
                destination_folder=FOLDER_FCGR,
                kmer=KMER,
                )

# Generate FCGR for a list of fasta files
generate_fcgr(list_fasta=LIST_FASTA,)

# count generated FCGR
N_seqs_to_gen = len(LIST_FASTA)
with open(FOLDER_FCGR.joinpath("generated_fcgr.txt"),"w") as fp:
    gen_seqs = len(list(FOLDER_FCGR.rglob("*.npy")))
    fp.write(f"{gen_seqs}/{N_seqs_to_gen}")