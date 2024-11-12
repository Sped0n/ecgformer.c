import numpy as np
import torch

from model import ECGformer

cache = {}


def extract(name: str):  # pyright: ignore[]
    def hook(model, input, output):  # pyright: ignore[]
        cache[name] = output.detach()  # pyright: ignore[]

    return hook  # pyright: ignore[]


m = ECGformer(
    signal_length=300,
    signal_channels=1,
    classes=5,
    embed_size=16,
    encoder_layers_num=2,
    encoder_heads=4,
    dropout=0.1,
)
m.load_state_dict(
    torch.load("./checkpoints/202411040353/model_55.pth", weights_only=True),
    strict=False,
)
m.eval()

m.embedding.register_forward_hook(extract("embedding"))  # pyright: ignore[]
m.encoder_layers[0].register_forward_hook(extract("encoder_layers.0"))  # pyright: ignore[]
m.encoder_layers[1].register_forward_hook(extract("encoder_layers.1"))  # pyright: ignore[]
m.classifier.register_forward_hook(extract("classifier"))  # pyright: ignore[]

print("Output")
input = np.loadtxt("../assets/input.txt")
print(m(torch.tensor(input).unsqueeze(0).reshape(1, 300, 1).float()))
print()

print("Embedding")
print(cache["embedding"].reshape(300, 16)[-1])  # pyright: ignore[]
print()

print("Encoder Layer 0")
print(cache["encoder_layers.0"].reshape(300, 16)[-1])  # pyright: ignore[]
print()

print("Encoder Layer 1")
print(cache["encoder_layers.1"].reshape(300, 16)[-1])  # pyright: ignore[]
print()

print("Classifier")
print(cache["classifier"])  # pyright: ignore[]
print()
