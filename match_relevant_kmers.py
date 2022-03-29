import sys 
import json 
import re
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from collections import namedtuple
from parameters import PARAMETERS

# inputs terminal
feature_from = sys.argv[1] # "saliency_map" or "shap_values"
max_rk = int(sys.argv[2]) # number of relevant kmers (for each clade) to consider

KMER = PARAMETERS["KMER"]
PATH_REFERENCE_GENOME = PARAMETERS["PATH_REFERENCE_GENOME"]
MatchMutation = namedtuple("Match", ["clade","mutation","kmer","pos_relevance","match_subseq","subseq"])

def find_matches(relevant_kmers, subseq, clade, mutation):
    "Find matches between a list of kmers and a subsequence"
    list_matches = []
    for j,kmer in enumerate(relevant_kmers):
        # find matches for each kmer in the mutated subsequence
        for match in re.finditer(kmer, subseq):
            if match:
                list_matches.append(
                    MatchMutation(clade, mutation, kmer, j+1, match.span()[0], subseq)
                )
    return list_matches

# load mutations
with open("mutations_reference.json") as fp: 
    mutations = json.load(fp)

# Reference sequence
ref_seq = next(SeqIO.parse(PATH_REFERENCE_GENOME, "fasta"))
seq = str(ref_seq.seq)

list_matches = []
CLADES=mutations.keys()
for clade in CLADES:
    print(clade)
    # iterator parameters
    path_rk = Path(f"data/{feature_from}/{clade}/relevant_kmers.csv")
    rk = pd.read_csv(path_rk)
    relevant_kmers=rk["kmer"].tolist()[:max_rk]
    list_mutations = mutations[clade]

    for mutation in list_mutations:
        char_matches = re.findall("\D+", mutation)
        num_matches  = re.findall("\d+", mutation)

        # update position to python format (-1) 
        positions = [int(num)-1 for num in num_matches]

        if "del" in char_matches: 
            print(f"deletion in position from {num_matches[0]} to {num_matches[1]}")
            # 1. replace mutation
            list_seq = list(seq)

            # 2. extract subsequence around the mutation based on KMER 
            # from position[0] to position[1] must be removed
            subseq_mutated = list_seq[(positions[0]-KMER+1):positions[0]] + list_seq[(positions[1]+1):(positions[1]+KMER)]
            subseq_mutated = "".join(subseq_mutated) # from list to string
            print(subseq_mutated)
            # 3. evalute if there is any match with relevant kmers
            matches_mutation = find_matches(relevant_kmers, subseq_mutated, clade, mutation)

            list_matches.extend(matches_mutation)

        elif len(char_matches)==1:
            print(f"change for many nucleotides in position {num_matches[0]}")

            list_seq = list(seq)
            if list_seq[positions[0]] == char_matches[0]:
                
                remaining_nucleotides = set(list("ACGT"))-set(char_matches[0])
                for nucleotide in remaining_nucleotides:
                    # 1. replace mutation
                    list_seq[positions[0]] = nucleotide

                    # 2. extract subsequence around the mutation based on KMER 
                    subseq_mutated = list_seq[(positions[0]-KMER+1):(positions[0]+KMER)]
                    subseq_mutated = "".join(subseq_mutated) # from list to string

                    # 3. evalute if there is any match with relevant kmers
                    matches_mutation = find_matches(relevant_kmers, subseq_mutated, clade, mutation)

                    list_matches.extend(matches_mutation)

        else:
            print(f"change on 1 nucleotide in position {num_matches[0]}")

            list_seq = list(seq)
            if list_seq[positions[0]] == char_matches[0]:
                
                # 1. replace mutation
                list_seq[positions[0]] = char_matches[1]

                # 2. extract subsequence around the mutation based on KMER 
                subseq_mutated = list_seq[(positions[0]-KMER+1):(positions[0]+KMER)]
                subseq_mutated = "".join(subseq_mutated) # from list to string

                # 3. evalute if there is any match with relevant kmers
                matches_mutation = find_matches(relevant_kmers, subseq_mutated, clade, mutation)

                list_matches.extend(matches_mutation)


# save results
path_save = Path(f"data/matches/{feature_from}")
path_save.mkdir(exist_ok=True, parents=True)
pd.DataFrame(list_matches).to_csv(path_save.joinpath(f"{KMER}mers.csv"))