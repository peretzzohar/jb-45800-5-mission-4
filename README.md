# Handwritten Letter Classifier

PyTorch project that trains an MLP on the **EMNIST Letters** dataset to recognize handwritten letters.

This final test was completed by Zohar Peretz at 20-08-2026.


### Files

* `train.py` — trains and evaluates the model
* `predict.py` — predicts a letter from an image
* `emnist01/` — EMNIST dataset
* `public/` — handwritten sample images created in **Microsoft Paint**
* `models/` — saved models
* `outputs/` — charts and prediction previews

### Setup

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -r requirements.txt
```

### Train

```bash
python train.py
```

Default: **25 epochs**.

Quick test:

```bash
python train.py --epochs 1 --batch-size 512
```

### Predict

```bash
python predict.py public/i-letter-2.png
```

The script predicts the letter, shows its confidence and top 3 guesses, and saves a preview.

### Dataset Credit

Uses the **EMNIST** dataset by Gregory Cohen, Saeed Afshar, Jonathan Tapson, and André van Schaik.
