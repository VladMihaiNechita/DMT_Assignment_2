# Final transformer pipeline

This folder contains the final submission code split into small files.

- `config.py`: paths and the fixed six-model bag settings.
- `data.py`: reads the train/test parquet files and builds the features.
- `features.py`: lists the columns used by the model.
- `preprocessing.py`: creates the engineered columns and encodes categories.
- `dataset.py`: groups rows by `srch_id` for packed transformer batches.
- `model.py`: the packed transformer ranker.
- `losses.py`: the ranking losses used during training.
- `train_model.py`: trains one transformer model.
- `predict.py`: scores the test set and writes `submission.csv`.
- `run.py`: runs the full bag end to end.

From the project root, run either command:

```bash
python -m Code.run
```

or:

```bash
python Code/run.py
```
