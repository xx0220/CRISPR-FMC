import os
import pickle
import numpy as np
import pandas as pd
import torch
import fm

rnafm_model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
rnafm_model.eval()

def onehot_encoding(seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([code_dict[base.upper()] for base in seq], dtype=np.float32).reshape(1, 23, 4)

def extract_rna_fm_features(seqs):
    rna_data = [(f"RNA_{i}", seq) for i, seq in enumerate(seqs)]
    _, _, batch_tokens = batch_converter(rna_data)
    with torch.no_grad():
        results = rnafm_model(batch_tokens, repr_layers=[12])
    embeddings = results["representations"][12]
    pooled = embeddings.mean(dim=1)
    return pooled.numpy()

def is_large_file_by_rows(file_path, row_threshold=10000):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i >= row_threshold


def process_large_csv(csv_path, output_pkl_path, chunk_size=1000):
    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    all_onehot = []
    all_rnafm = []
    all_labels = []

    for chunk_id, chunk in enumerate(reader):

        sequences = chunk['sgRNA'].tolist()
        labels = chunk['indel'].values.astype(np.float32)

        X_onehot = np.array([onehot_encoding(seq) for seq in sequences], dtype=np.float32).reshape(len(sequences), 1, 23, 4)
        X_rnafm = extract_rna_fm_features(sequences)

        all_onehot.append(X_onehot)
        all_rnafm.append(X_rnafm)
        all_labels.append(labels)

    # 拼接所有批次
    final_onehot = np.concatenate(all_onehot, axis=0)
    final_rnafm = np.concatenate(all_rnafm, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)

    with open(output_pkl_path, 'wb') as f:
        pickle.dump((final_onehot, final_rnafm, final_labels), f)


def process_dataset(csv_path, output_path):
    df = pd.read_csv(csv_path)
    sequences = df['sgRNA'].tolist()
    labels = df['indel'].values.astype(np.float32)

    X_onehot = np.array([onehot_encoding(seq) for seq in sequences], dtype=np.float32).reshape(len(sequences), 1, 23, 4)
    print(len(sequences))
    X_rnafm = extract_rna_fm_features(sequences)

    with open(output_path, 'wb') as f:
        pickle.dump((X_onehot, X_rnafm, labels), f)

def chuli(sequence):

    X_onehot = np.array([onehot_encoding(sequence)], dtype=np.float32).reshape(1, 1, 23, 4)
    X_rnafm = extract_rna_fm_features([sequence])

    return X_onehot, X_rnafm



if __name__ == "__main__":
    seq = "AAAAAACAGATGCCACCTGTGGG"
    print(chuli(seq))