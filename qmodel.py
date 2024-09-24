import torch
import torch.nn as nn
from torch import Tensor

import brevitas.nn as qnn


class QuantEmbedding(nn.Module):
    def __init__(self, signal_channels: int, embed_size: int):
        super().__init__()
        self.linear = nn.Linear(signal_channels, embed_size)
        self.lieanr1 = qnn.QuantLinear(
            signal_channels,
            embed_size,
        )
        self.norm = nn.LayerNorm(embed_size)
        self.activation = qnn.QuantReLU()

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class QuantMLP(nn.Module):
    def __init__(self, embed_size: int, expansion: int):
        super().__init__()
        self.linear1 = qnn.QuantLinear(embed_size, embed_size * expansion)
        self.activation = qnn.QuantReLU()
        self.linear2 = qnn.QuantLinear(embed_size * expansion, embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class QuantClassifier(nn.Module):
    def __init__(self, embed_size: int, classes: int):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.linear1 = qnn.QuantLinear(embed_size, classes)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = torch.mean(x, dim=1)
        x = self.norm(x)
        x = self.linear1(x)
        return x


class QuantEncoderLayer(nn.Module):
    def __init__(self, embed_size: int, heads: int, expansion: int, dropout: float):
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embed_size = embed_size
        self.attention = qnn.QuantMultiheadAttention(
            embed_size, heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_size)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp = QuantMLP(embed_size, expansion)
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


class QuantECGformer(nn.Module):
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
        # TODO: use QuantIdentity to quantize input
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.embedding = QuantEmbedding(signal_channels, embed_size)
        self.encoder_layers = nn.ModuleList(
            [
                QuantEncoderLayer(embed_size, encoder_heads, mlp_expansion, dropout)
                for _ in range(encoder_layers_num)
            ]
        )
        self.classifier = QuantClassifier(embed_size, classes)
        self.positional_embedding = qnn.QuantEmbedding(signal_length, embed_size)

    def forward(self, x: Tensor) -> Tensor:  # pyright: ignore[reportImplicitOverride]
        x = self.embedding(x)

        positions = (
            torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        )
        pos_embed = self.positional_embedding(positions)
        x = x + pos_embed

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        x = self.classifier(x)
        return x
