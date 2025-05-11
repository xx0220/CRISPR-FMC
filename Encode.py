import os
import pickle
import torch
import pandas as pd
import fairseq

# ====== 加载 RNA-FM 模型 ======
rnafm_model, alphabet = fairseq.models.roberta.RobertaModel.from_pretrained(
    model_name_or_path='./rna_fm_t12',
    checkpoint_file='model.pt',
    data_name_or_path='./rna_fm_t12'
), None
batch_converter = rnafm_model.task.source_dictionary
rnafm_model.eval()


# ====== 编码函数们 ======
def onehot_encoding(seq):
    code_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return torch.tensor([code_dict[base.upper()] for base in seq], dtype=torch.float32).reshape(1, 23, 4)


def extract_rna_fm_features(seqs):
    rna_data = [(f"RNA_{i}", seq) for i, seq in enumerate(seqs)]
    tokens = []
    for name, seq in rna_data:
        token = torch.tensor([batch_converter.index(c) for c in list(seq.upper())])
        tokens.append(token)
    batch_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=1)  # pad with <pad> token
    with torch.no_grad():
        results = rnafm_model.model(batch_tokens)
        embeddings = results[0]  # (batch_size, seq_len, hidden_size)
        pooled = embeddings.mean(dim=1)  # (batch_size, hidden_size)
        pooled = pooled[:, :640]  # 截取前640维
    return pooled  # Tensor


def is_large_file_by_rows(file_path, row_threshold=10000):
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i >= row_threshold


# ====== 处理大文件（分批） ======
def process_large_csv(csv_path, output_pkl_path, chunk_size=1000):
    reader = pd.read_csv(csv_path, chunksize=chunk_size)

    all_onehot = []
    all_rnafm = []
    all_labels = []
    all_seqs = []

    for chunk_id, chunk in enumerate(reader):
        sequences = chunk['sgRNA'].tolist()
        labels = torch.tensor(chunk['indel'].values, dtype=torch.float32)

        X_onehot = torch.stack([onehot_encoding(seq) for seq in sequences])  # (batch, 1, 23, 4)
        X_rnafm = extract_rna_fm_features(sequences)  # (batch, 640)

        all_onehot.append(X_onehot)
        all_rnafm.append(X_rnafm)
        all_labels.append(labels)
        all_seqs.extend(sequences)

    final_onehot = torch.cat(all_onehot, dim=0)  # (N, 1, 23, 4)
    final_rnafm = torch.cat(all_rnafm, dim=0)  # (N, 640)
    final_labels = torch.cat(all_labels, dim=0)  # (N, )

    with open(output_pkl_path, 'wb') as f:
        pickle.dump((final_onehot, final_rnafm, final_labels, all_seqs), f)

def process_dataset(csv_path, output_path):
    df = pd.read_csv(csv_path)
    sequences = df['sgRNA'].tolist()
    labels = torch.tensor(df['indel'].values, dtype=torch.float32)

    X_onehot = torch.stack([onehot_encoding(seq) for seq in sequences])
    X_rnafm = extract_rna_fm_features(sequences)

    with open(output_path, 'wb') as f:
        pickle.dump((X_onehot, X_rnafm, labels, sequences), f)


# ====== 主程序入口 ======
if __name__ == "__main__":
    input_dir = './datasets'
    output_dir = './processed_data'
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            csv_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.pkl")

            if is_large_file_by_rows(csv_path):
                process_large_csv(csv_path, output_path, chunk_size=1000)
            else:
                process_dataset(csv_path, output_path)
