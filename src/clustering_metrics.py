import pandas as pd
import numpy as np 
from itertools import combinations

class GeneralizedDiscriminationValue: 

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, embeddings, predictions): 
        """Compute Generalized Discrimination Value

        Args:
            embeddings (numpy.ndarray): D-dimensional array, one row per each sample
            labels (list): label (one) for each sample, same order than embeddings
        """ 

        # z-scored each dimension       
        mu  = np.mean(embeddings,axis=1)
        std = np.std(embeddings, axis=1)
        
        # mimic dimension of embeddings
        mu_outer = np.outer(mu, np.ones(embeddings.shape[1]))
        std_outer = np.outer(std, np.ones(embeddings.shape[1]))

        # compute z-scored embeddings as proposed in the paper
        S = 0.5*(embeddings-mu_outer) / std_outer

        # mean intra class
        m_intra = self.mean_intra_class(S, predictions)

        # mean inter class
        m_inter = self.mean_inter_class(S, predictions)

        # GDV
        D = embeddings.shape[1]
        L = len(self.classes)

        sum_mean_intra_class = sum(m_intra.values())
        sum_mean_inter_class = sum(m_inter.values())

        gdv = 1/np.sqrt(D) * ( 1/L*sum_mean_intra_class - 2/(L*(L-1))*sum_mean_inter_class )
        
        return gdv

    def mean_intra_class(self, z_score_embeddings, predictions):
        CLADES = self.classes
        # mean_intra_class distances for each class
        mean_intra_class = dict()

        for clade in CLADES:
            idx_class = predictions.query(f"prediction == '{clade}'").index.tolist()

            sum_d = 0
            N = len(idx_class)

            for pos, i in enumerate(idx_class[:(N-1)]): 
                for j in idx_class[(pos+1):]:
                    sum_d += np.linalg.norm(z_score_embeddings[i,:] - z_score_embeddings[j,:])

            mean_icd = 2/(N*(N-1)) * sum_d
            
            mean_intra_class[clade] = mean_icd

        return mean_intra_class

    def mean_inter_class(self, z_score_embeddings, predictions):
        CLADES = self.classes
        # mean inter class distances for each pair of classes
        mean_inter_class = dict()

        for clade1, clade2 in combinations(CLADES,2):
            idx_class1 = predictions.query(f"prediction == '{clade1}'").index.tolist()
            idx_class2 = predictions.query(f"prediction == '{clade2}'").index.tolist()
            
            sum_d = 0
            N1 = len(idx_class1) 
            N2 = len(idx_class2)
            
            for i in idx_class1: 
                for j in idx_class2:
                    sum_d += np.linalg.norm(z_score_embeddings[i,:] - z_score_embeddings[j,:])
            
            mean_icd = 1/(N1*N2) * sum_d
            
            mean_inter_class[(clade1,clade2)] = mean_icd

        return mean_inter_class