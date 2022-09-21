import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import cycle 
from collections import namedtuple

from sklearn.metrics import (
    precision_recall_curve, # must be applied to each class independently
    average_precision_score,
    precision_recall_fscore_support, # retrieve precision and recall
    accuracy_score,
    PrecisionRecallDisplay,
    confusion_matrix,
    roc_auc_score, # use multiclass='ovo' 
)
from src.metrics.matthews_corrcoef import mcc
from parameters import PARAMETERS

KFOLD=sys.argv[-1]

# load predictions
CLADES = PARAMETERS["CLADES"]
predictions = pd.read_csv(f"data/test-{KFOLD}/predictions.csv")
conf  = predictions.confidence # confidence: probability given by argmax over the output of the model
preds = predictions.prediction # predictions: clade with the highest confidence
gt    = predictions.ground_truth # ground-truth: real clade

# metrics precision and recall
precision, recall, fscore, support = precision_recall_fscore_support(
                                    y_true=gt, 
                                    y_pred=preds, 
                                    average=None, 
                                    labels=CLADES,
                                    zero_division=0
                                    )

list_metrics = []
Metrics = namedtuple("Metrics", ["clade","precision", "recall", "fscore", "support"])
for j,clade in enumerate(CLADES): 
    list_metrics.append(
        Metrics(clade, precision[j], recall[j], fscore[j], support[j])
    )

df_metrics = pd.DataFrame(list_metrics)
df_metrics.to_csv(f"data/test-{KFOLD}/metrics.csv")

# accuracy
accuracy = accuracy_score( 
    y_true = gt,
    y_pred = preds
)

# Matthews coefficient
cm = confusion_matrix(y_true=gt, y_pred=preds)
m_coeff = mcc(cm)

# load probabilities
embeddings = np.load(f"data/test-{KFOLD}/embeddings.npy")

# auc score
gt_hotencode = np.zeros((len(gt), len(CLADES)))
for j,y in enumerate(gt):
    col = CLADES.index(y)
    gt_hotencode[j,col] = 1
auc = roc_auc_score(y_true=gt_hotencode, y_pred=embeddings, multi_class='ovo')

with open(f"data/test-{KFOLD}/global_metrics.json","w") as fp:
    json.dump(
        {
            "acc": accuracy,
            "mcc": m_coeff,
            "auc": auc
        },
        fp,
        indent=4
    )

## Curve Precision Recall

# Compute precision recall curve for each clade
precision = dict()
recall = dict()
average_precision = dict()
for j,clade in enumerate(CLADES):
    y_true = predictions.ground_truth
    y_score = embeddings[:,j] # probabilities for the clade
    precision[clade], recall[clade], _ = precision_recall_curve(y_true, y_score, pos_label = clade)

# -----------------------------------

# Plot
linestyle_tuple = [
#      ('loosely dotted',        (0, (1, 10))),
#      ('dotted',                (0, (1, 1))),
#      ('densely dotted',        (0, (1, 1))),

     ('loosely dashed',        (0, (5, 10))),
     ('dashed',                (0, (5, 5))),
     ('densely dashed',        (0, (5, 1))),

     ('loosely dashdotted',    (0, (3, 10, 1, 10))),
     ('dashdotted',            (0, (3, 5, 1, 5))),
     ('densely dashdotted',    (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
]
colors = cycle(["navy", "turquoise", "darkorange", "cornflowerblue", "teal"])

_, ax = plt.subplots(figsize=(7,8))

for i, color, linestyle in zip(CLADES, colors, linestyle_tuple):
    display = PrecisionRecallDisplay(
        recall=recall[i],
        precision=precision[i],
    )
    display.plot(ax=ax, name=f"Precision-recall for clade {i}", color=color, linestyle = linestyle[1])
plt.savefig(f"data/test-{KFOLD}/curve_pr.jpg")
plt.savefig(f"data/test-{KFOLD}/curve_pr.pdf")