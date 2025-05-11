# CRISPR-FMC

A dual-branch hybrid deep learning model for CRISPR/Cas9 sgRNA on-target activity prediction. CRISPR-FMC integrates one-hot encoding and RNA-FM embeddings, combined with MSC, Transformer, BiGRU, and bidirectional cross-attention modules.

---

## Model Architecture Overview

The architecture includes:
* One-hot and RNA-FM dual-branch encoding
* Multi-Scale Convolution (MSC) modules for local pattern extraction
* Transformer encoders and BiGRU for global context modeling
* Bidirectional cross-attention and residual FFN for multimodal fusion
* A final MLP for regression-based on-target activity prediction

---

## Environment

* Python 3.9
* PyTorch 1.8.1 + cu101
* fair-esm 2.0.0
* pandas 2.2.3
* numpy 1.26.4
* matplotlib 3.9.4
* scikit-learn 1.6.1
* tqdm 4.67.1
* flask 3.1.0 (optional for web demo)
* torchvision 0.9.1 + cu101
* torchaudio 0.8.1

---

## Datasets

This project uses 9 publicly available CRISPR-Cas9 on-target efficiency datasets:

* WT
* ESP
* HF
* xCas
* SpCas9-NG
* Sniper-Cas9
* HCT116
* HELA
* HL60

Each dataset contains 23-nt sgRNA sequences and normalized indel values.

---

## File Description

* `CRISPR_FMC_model.py`: Defines the CRISPR-FMC model architecture.
* `CRISPR_FMC_train.py`: Trains and evaluates the model using 5-fold cross-validation on all datasets.
* `Encode.py`: Preprocesses .csv datasets and extracts One-hot and RNA-FM features into `.pkl` files.
* `datasets/`: Raw `.csv` files with `sgRNA` and `indel` columns.

---

## How to Use

### 1. Preprocess datasets

Place raw .csv files (with `sgRNA`, `indel` columns) in `./datasets/`, then run:

```bash
python Encode.py
```

### 2. Train and evaluate model

```bash
python CRISPR_FMC_train.py
```

Model checkpoints and evaluation results will be saved automatically.

---

## Results & Metrics

Each dataset is evaluated via 5-fold cross-validation using:
* SCC (Spearman correlation coefficient)
* PCC (Pearson correlation coefficient)


---

