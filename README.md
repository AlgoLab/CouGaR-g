# fcgr-cnn
Frequence Chaos Game Representation with Deep Learning

Create a virtual environment and install packages
```
python -m venv fcgr
source fcgr/bin/activate
pip install -r requirements.txt
```

Set parameters for the experiment in `parameters.py`
- See (and include) preprocessing functions at `preprocessing.py`

Run codes in this order
1. `undersample_sequences.py`
2. `extract_sequences.py`
3. `split_data.py` will create a file `datasets.json` with train, validation and test sets
4. `train.py`
5. `test.py`
6. `classification_metrics` 
7. `clustering_metrics` 

A folder `data/` will be created after training. 
- `data/checkpoints` will save the best weights during training.
- `data/test` will save all the metrics (classification and clustering) resulting from the evaluation of the best model on the test set.