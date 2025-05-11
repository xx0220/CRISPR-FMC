# BiGRU
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, embed_size):
        super(MultiHeadAttention, self).__init__()
        self.head_num = head_num
        self.embed_size = embed_size
        self.head_dim = embed_size // head_num
        assert self.head_dim * head_num == embed_size, "embed_size must be divisible by head_num"

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        query = self.queries(query)

        values = values.view(N, value_len, self.head_num, self.head_dim).transpose(1, 2)
        keys = keys.view(N, key_len, self.head_num, self.head_dim).transpose(1, 2)
        query = query.view(N, query_len, self.head_num, self.head_dim).transpose(1, 2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])
        attention = torch.softmax(energy / (self.head_dim ** 0.5), dim=-1)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        out = out.reshape(N, query_len, self.head_num * self.head_dim)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, head_num, ffn_dim):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(head_num, embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, ffn_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ffn_dim, embed_size)
        )

    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class BiGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiGRULayer, self).__init__()
        self.bigru = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, _ = self.bigru(x)
        return out


class CRISPR_FMC(nn.Module):
    def __init__(self, embed_size=64, head_num=4, ffn_dim=512, dropout=0.2):
        super(CRISPR_FMC, self).__init__()

        self.onehot_conv1 = nn.Conv2d(4, 16, kernel_size=(1, 1))
        self.onehot_conv3 = nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1))
        self.onehot_conv5 = nn.Conv2d(4, 16, kernel_size=(5, 5), padding=(2, 2))
        self.onehot_conv7 = nn.Conv2d(4, 16, kernel_size=(7, 7), padding=(3, 3))

        self.transformer1 = TransformerBlock(embed_size, head_num, ffn_dim)
        self.bigru1 = BiGRULayer(embed_size, 128)

        self.rnafm_dense = nn.Linear(640, 23 * embed_size)
        self.transformer2 = TransformerBlock(embed_size, head_num, ffn_dim)
        self.bigru2 = BiGRULayer(embed_size, 128)

        self.attn1 = MultiHeadAttention(head_num, embed_size)
        self.attn2 = MultiHeadAttention(head_num, embed_size)
        self.attn_drop1 = nn.Dropout(dropout)
        self.attn_drop2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.x1_proj = nn.Linear(256, embed_size)
        self.x2_proj = nn.Linear(256, embed_size)

        self.residual_ffn = nn.Sequential(
            nn.Linear(2 * embed_size, 2 * embed_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2 * embed_size, 2 * embed_size)
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 23, 80),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

    def forward(self, onehot_input, rnafm_input):
        # expect: onehot_input: [B, 4, 23, 1]
        x1 = torch.cat([
            F.relu(self.onehot_conv1(onehot_input)),
            F.relu(self.onehot_conv3(onehot_input)),
            F.relu(self.onehot_conv5(onehot_input)),
            F.relu(self.onehot_conv7(onehot_input))
        ], dim=1)
        x1 = x1.squeeze(-1).permute(0, 2, 1)
        x1 = self.transformer1(x1)
        x1 = self.bigru1(x1)

        x2 = F.relu(self.rnafm_dense(rnafm_input))
        x2 = x2.view(-1, 23, self.transformer2.attention.embed_size)
        x2 = self.transformer2(x2)
        x2 = self.bigru2(x2)

        x1 = self.x1_proj(x1)
        x2 = self.x2_proj(x2)

        x1_to_x2 = self.attn1(x1, x2, x2)
        x1_to_x2 = self.norm1(x1 + x1_to_x2)
        x1_to_x2 = self.attn_drop1(x1_to_x2)

        x2_to_x1 = self.attn2(x2, x1, x1)
        x2_to_x1 = self.norm2(x2 + x2_to_x1)
        x2_to_x1 = self.attn_drop2(x2_to_x1)

        residual = torch.cat([x1_to_x2, x2_to_x1], dim=-1)
        merged = self.residual_ffn(residual) + residual

        merged = merged.view(merged.size(0), -1)
        out = self.fc(merged)
        return out
