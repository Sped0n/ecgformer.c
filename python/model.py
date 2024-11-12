from math import sqrt

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, signal_channels: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(signal_channels, embed_size)
        self.norm = nn.LayerNorm(embed_size, elementwise_affine=False, bias=False)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        self.queries_projection = nn.Linear(embed_size, embed_size)
        self.values_projection = nn.Linear(embed_size, embed_size)
        self.keys_projection = nn.Linear(embed_size, embed_size)
        self.final_projection = nn.Linear(embed_size, embed_size)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.divider = int(sqrt(self.head_dim))

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        batch_size, seq_len, _ = x.shape

        # Linear projections
        keys = self.keys_projection(x)
        values = self.values_projection(x)
        queries = self.queries_projection(x)

        # Reshape to separate heads
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention computation
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores
        energy_term = torch.matmul(queries, keys.transpose(-2, -1)) / self.divider

        # Apply softmax
        attention = torch.softmax(energy_term, dim=-1)

        # Apply attention to values
        out = torch.matmul(attention, values)

        # Reshape and transpose back
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.embed_size)

        return self.final_projection(out)


class MLP(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, embed_size)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embed_size, embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embed_size: int, classes: int):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.linear = nn.Linear(embed_size, classes)
        self.norm = nn.LayerNorm(embed_size, elementwise_affine=False, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.linear(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embed_size = embed_size
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size, elementwise_affine=False, bias=False)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(embed_size)
        self.norm2 = nn.LayerNorm(embed_size, elementwise_affine=False, bias=False)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout1(x)
        x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.dropout2(x)
        x += residual

        return x


class ECGformer(nn.Module):
    def __init__(
        self,
        signal_length: int,
        signal_channels: int,
        classes: int,
        embed_size: int,
        encoder_layers_num: int,
        encoder_heads: int,
        dropout: float,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embedding = Embedding(signal_channels, embed_size)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, encoder_heads, dropout)
                for _ in range(encoder_layers_num)
            ]
        )
        self.classifier = Classifier(embed_size, classes)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.embedding(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.classifier(x)
        return x
