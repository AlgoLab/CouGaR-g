"""
This script assumes that each Fasta file contains only one sequence
"""
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
import numpy as np 
from .fcgr import FCGR

class GenerateFCGR: 

    def __init__(self, destination_folder: Path = "img", kmer: int = 8, ): 
        self.destination_folder = Path(destination_folder)
        self.kmer = kmer
        self.fcgr = FCGR(kmer)
        self.counter = 0 # count number of time a sequence is converted to fcgr

        # Create destination folder if needed
        self.destination_folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, list_fasta,):
         
        for fasta in tqdm(list_fasta, desc="Generating FCGR"):
            self.from_fasta(fasta)

    def from_fasta(self, path: Path,):
        """FCGR for a sequence in a fasta file.
        The FCGR image will be save in 'destination_folder/specie/label/id_fasta.jpg'
        """
        # load fasta file
        path = Path(path)
        fasta  = self.load_fasta(path)
        record = next(fasta)
                
        # Generate and save FCGR for the current sequence
        _, specie, label  = str(path.parents[0]).split("/")
        id_fasta = path.stem
        path_save = self.destination_folder.joinpath("{}/{}/{}.npy".format(specie, label, id_fasta))
        path_save.parents[0].mkdir(parents=True, exist_ok=True)
        self.from_seq(str(record.seq), path_save)
        
    def from_seq(self, seq: str, path_save):
        "Get FCGR from a sequence"
        if not Path(path_save).is_file():
            seq = self.preprocessing(seq)
            chaos = self.fcgr(seq)
            np.save(path_save, chaos)
        self.counter +=1

    def reset_counter(self,):
        self.counter=0
        
    @staticmethod
    def preprocessing(seq):
        seq = seq.upper()
        for letter in "BDEFHIJKLMOPQRSUVWXYZ":
            seq = seq.replace(letter,"N")
        return seq

    @staticmethod
    def load_fasta(path: Path):
        return SeqIO.parse(path, "fasta")