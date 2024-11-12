from pathlib import Path

import numpy as np
import onnx
import torch
from numpy.typing import NDArray
from onnxsim import simplify  # pyright: ignore[]

from model import ECGformer


def save_tensor_to_txt(tensor: torch.Tensor, filename: str | Path, fmt: str = ".6f"):
    array: NDArray[np.float32] = tensor.detach().cpu().numpy().transpose()

    if len(array.shape) == 1:
        numbers = [f"{x:{fmt}}" for x in array]
        line = " ".join(numbers)
        with open(filename, "w") as f:
            f.write(line)
    elif len(array.shape) == 2:
        with open(filename, "w") as f:
            for row in array:
                numbers = [f"{x:{fmt}}" for x in row]
                line = " ".join(numbers) + "\n"
                f.write(line)


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

total = sum(p.numel() for p in m.parameters())
trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"Total: {total}, Trainable: {trainable}")

torch.onnx.export(m, (torch.randn(1, 300, 1),), "ecgformer.onnx")
om = onnx.load("ecgformer.onnx")
simplified, ok = simplify(om)
if not ok:
    raise Exception("Simplification failed")
onnx.save(simplified, "ecgformer_simplified.onnx")

param_dir = Path("../assets/params")
if not param_dir.exists():
    param_dir.mkdir()
else:
    for f in param_dir.glob("*.txt"):
        f.unlink()
for n, p in m.named_parameters():
    print(n, ": ", p.size())
    save_tensor_to_txt(p, param_dir.joinpath(f"{n}.txt"))
