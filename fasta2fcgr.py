from pathlib import Path
from parameters import PARAMETERS
from src.generate_fcgr import GenerateFCGR

FOLDER_FASTA = Path(PARAMETERS["FOLDER_FASTA"]) 
LIST_FASTA   = list(FOLDER_FASTA.rglob("*fasta"))
KMER = PARAMETERS["KMER"] 
FOLDER_FCGR = PARAMETERS["FOLDER_FCGR"]
#Instantiate class to generate FCGR
generate_fcgr = GenerateFCGR(
                destination_folder=FOLDER_FCGR,
                kmer=KMER,
                )

# Generate FCGR for a list of fasta files
generate_fcgr(list_fasta=LIST_FASTA,)