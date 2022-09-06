
import re
import numpy as np 

from collections import namedtuple
from itertools import product
from PIL import Image

from .fcgr import FCGR

def array2img(array):
    "FCGR array to grayscale image"
    max_color = 255
    m, M = array.min(), array.max()
    # rescale to [0,1]
    img_rescaled = (array - m) / (M-m) 

    # invert colors black->white
    img_array = np.ceil(max_color - img_rescaled*max_color)
    img_array = np.array(img_array, dtype=np.int8)

    # convert to Image 
    img_pil = Image.fromarray(img_array,'L')
    return img_pil

def find_matches(kmer: int, seq: str, return_str: bool=False):
    "Find all the started positions where the kmer appears in the sequence"
    idx_matches = []
    for match in re.finditer(kmer, seq):
        idx_matches.append(match.span()[0])

    if return_str:
        idx_matches = ",".join(str(_) for _ in idx_matches) if isinstance(idx_matches,list) else str(idx_matches)
    return idx_matches

def fcgrpos2kmers(k: int):
    """build a dictionary with {(px,py): kmer} for all 2**k kmers
    where px and py are the positions in the FCGR array of the kmer
    """
    fcgr = FCGR(k)  
    
    # list of all kmers
    list_kmers = list("".join(elem) for elem in product("ACGT", repeat=k))

    def kmer2pos(kmer):
        px,py=fcgr.pixel_position(kmer)
        return px-1,py-1 # to start in 0 instead of 1

    dict_kmer2pos = {kmer: kmer2pos(kmer) for kmer in list_kmers}
    return {pos: kmer for kmer,pos in dict_kmer2pos.items()}

def preprocess_seq(seq):
    "Remove all characters different from A,C,G,T or N"
    seq = seq.upper()
    for letter in "BDEFHIJKLMOPQRSUVWXYZ":
        seq = seq.replace(letter,"N")
    return seq

def kmer_importance(smap, threshold, fcgr, fcgrpos2kmer):
    """find most relevant kmers for the prediction based on saliency map

    Args:
        smap (array]): saliency map
        threshold (float): between 0 and 1 to filter gradients in smap
        fcgr (array): the FCGR matrix
        pos2kmer (dict): dictionary to map position in FCGR to the kmer

    Returns:
        list: information about kmer importance, sorted by the value of the gradient
    """    
    kmer_importance=namedtuple("kmer_importance", ["kmer","row","col","grad","freq"])

    rows, cols = np.where(smap > threshold)
    list_kmers = []
    for row,col in zip(rows,cols):
        list_kmers.append(
            kmer_importance(fcgrpos2kmer.get((row,col)), 
            row,
            col, 
            smap[row,col], 
            fcgr[row,col]
            )
        )

    # Sort values by importance (value of the gradient) 
    list_kmers = sorted(list_kmers, key=lambda x: x.grad,reverse=True)

    return list_kmers


def kmer_importance_shap(shap_values, threshold, fcgr, fcgrpos2kmer, Nmin = None):
    """find most relevant kmers for the prediction based on saliency map
    Args:
        shap_values (array): shap_values for the clade of the fcgr
        threshold (float): to filter elements in abs(shap_values)
        fcgr (array): the FCGR matrix
        fcgrpos2kmer (dict): dictionary to map position in FCGR to the kmer
    Returns:
        list: information about kmer importance, sorted by the value of the gradient
    """    
    kmer_importance=namedtuple("kmer_importance", ["kmer","row","col","sv","freq"])

    rows, cols = np.where(np.abs(shap_values) > threshold)
    
    # if number of elements is less than the minimum required, analyze all elements
    if Nmin is not None:
        if len(rows) < Nmin: 
            rows, cols = np.where(np.abs(shap_values) > 0.)

    list_kmers = []
    for row,col in zip(rows,cols):
        list_kmers.append(
            kmer_importance(fcgrpos2kmer.get((row,col)), 
            row,
            col, 
            shap_values[row,col], 
            fcgr[row,col]
            )
        )
    
    # Sort values by importance (value of the gradient) 
    list_kmers = sorted(list_kmers, key=lambda x: abs(x.sv),reverse=True)
    return list_kmers