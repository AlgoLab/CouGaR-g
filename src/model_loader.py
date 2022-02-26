"""Load model from /models"""
import importlib
import os

from pathlib import Path
from typing import Optional

from tensorflow.python.eager.context import num_gpus

OMMIT = {".ipynb_checkpoints","__pycache__","__init__","custom_layers","custom_losses"} # files to be ommited
BASE_DIR = Path(__file__).resolve().parent # base directory unsupervised-dna
BASE_MODELS = BASE_DIR.joinpath("models") # models directory

class ModelLoader:
    "Load models for unsupervised learning using FCGR (grayscale images)"

    AVAILABLE_MODELS = [model[:-3] for model in os.listdir(BASE_MODELS) if all([ommit not in model for ommit in OMMIT])]

    def __call__(self, model_name: str, n_outputs: int, weights_path: Optional[Path]=None):
        "Get keras model"
        
        # Call class of model to load
        get_model = getattr(
            importlib.import_module(
                f"supervised_dna.models.{model_name}"
            ),
            "get_model")        
        
        # Load architecture
        model = get_model(n_outputs)
    
        # Load weights to the model from file
        if weights_path is not None:
            print(f"\n **load model weights_path** : {weights_path}")
            model.load_weights(weights_path)

        print("\n**Model created**")        

        return model