import pandas as pd
import numpy as np 
from itertools import combinations

from sklearn.metrics import (
    silhouette_score,
    #silhouette_samples,
    calinski_harabasz_score,
)
from metrics.clustering_metrics import GeneralizedDiscriminationValue as GDV

from parameters import PARAMETERS
CLADES = PARAMETERS["CLADES"]

predictions = pd.read_csv("data/test/predictions.csv")
embeddings  = np.load("data/test/embeddings.npy")

labels = predictions.prediction

# Global Calinski Harabasz Score
global_calinski_harabasz = calinski_harabasz_score(X=embeddings, labels = labels)

# Global silhouette score
global_silhouette = silhouette_score(X=embeddings, labels=labels)

# Generalized Discrimination Value
gdv = GDV(classes=CLADES)
global_gdv = gdv(embeddings, predictions)

clust_metrics = {
    "silhouette": global_silhouette,
    "calinski_harabasz": global_calinski_harabasz,
    "GDV": global_gdv
}
pd.DataFrame.from_dict(clust_metrics, orient='index').T.to_csv("data/test/clustering_metrics.csv")