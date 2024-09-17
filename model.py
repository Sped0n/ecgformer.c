import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, signal_channels: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(signal_channels, embed_size)
        self.norm = nn.LayerNorm(embed_size)
        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MLP(nn.Module):
    def __init__(self, embed_size: int, expansion: int):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, embed_size * expansion)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(embed_size * expansion, embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class Classifier(nn.Module):
    def __init__(self, embed_size: int, classes: int):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.linear1 = nn.Linear(embed_size, classes)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.linear1(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size: int, heads: int, expansion: int, dropout: float):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embed_size = embed_size
        self.attention = nn.MultiheadAttention(embed_size, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = MLP(embed_size, expansion)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x)[0]
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
        mlp_expansion: int,
        dropout: float,
    ):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embedding = Embedding(signal_channels, embed_size)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_size, encoder_heads, mlp_expansion, dropout)
                for _ in range(encoder_layers_num)
            ]
        )
        self.classifier = Classifier(embed_size, classes)
        self.positional_embedding = nn.Parameter(torch.randn(signal_length, embed_size))

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.embedding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x + self.positional_embedding)
        x = self.classifier(x)
        return x
