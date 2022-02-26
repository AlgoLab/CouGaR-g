import numpy as np 

class GeneralizedDiscriminationValue: 

    def __init__(self,):
        pass

    def __call__(self, embeddings, labels): 
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
        m_intra = self.mean_intra_class(embeddings, labels)

        # mean inter class
        m_inter = self.mean_inter_class(embeddings, labels)

        
    def mean_intra_class(self,):
        pass

    def mean_inter_class(self,):
        pass

    