"""
Plot accuracy and loss for train/val
Confusion matrix
"""
from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.metrics import ConfusionMatrixDisplay

from parameters import PARAMETERS
KMER = PARAMETERS["KMER"]
PATH = Path("data")
PATH_PLOTS = PATH.joinpath("plots")
PATH_PLOTS.mkdir(exist_ok=True, parents=True)

## Training metrics
training_log = pd.read_csv(PATH.joinpath("train/training_log.csv"))

# Accuracy
training_log.rename({
                    "accuracy": "training", 
                    "val_accuracy": "validation",
                    },
                    axis=1,
                    ).plot("epoch", ["training", "validation"], 
                            title = f"Model Accuracy | kmer={KMER}")
plt.savefig(PATH_PLOTS.joinpath(f"accuracy_{KMER}mer.jpg"))
plt.savefig(PATH_PLOTS.joinpath(f"accuracy_{KMER}mer.pdf"))
# Loss
training_log.rename({
                    "loss": "training",
                    "val_loss": "validation"
                    },
                    axis=1,
                    ).plot("epoch", ["training", "validation"], 
                            title=f"Model Loss | kmer={KMER}")
plt.savefig(PATH_PLOTS.joinpath(f"loss_{KMER}mer.jpg"))
plt.savefig(PATH_PLOTS.joinpath(f"loss_{KMER}mer.pdf"))

## Confusion matrix
predictions = pd.read_csv(PATH.joinpath("test/predictions.csv"))
y_true = predictions.ground_truth
y_pred = predictions.prediction
ConfusionMatrixDisplay.from_predictions(y_true, y_pred,)
plt.title(f"Confusion matrix | kmer={KMER}")
plt.savefig(PATH_PLOTS.joinpath(f"confusion_matrix_{KMER}mer.jpg"))
plt.savefig(PATH_PLOTS.joinpath(f"confusion_matrix_{KMER}mer.pdf"))

