import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from config import Config
from model import ECGformer


class NPYDataset(Dataset[Any]):
    def __init__(self, targets: list[str]) -> None:
        super().__init__()
        data_cache: NDArray[np.float32] | None = None
        labels_cache: list[int] = []
        for index, target in enumerate(targets):
            data: NDArray[np.float32] = np.load(f"../assets/dataset/{target}.npy")
            if data_cache is None:
                data_cache = data
            else:
                data_cache = np.vstack((data_cache, data))
            labels_cache += [index for _ in range(data.shape[0])]
        self.__data: Tensor = torch.from_numpy(data_cache).float()
        self.__labels: Tensor = torch.tensor(labels_cache).long()

    def __len__(self) -> int:
        return len(self.__labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.__data[index].unsqueeze(0), self.__labels[index]


class Trainer:
    def __init__(self, config: Config, dataset: NPYDataset) -> None:
        # data preparation
        train_size = int(config.train_proportion * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        self.__batch_size = config.dl_batch_size
        self.__train_loader = DataLoader(
            train_dataset,
            batch_size=self.__batch_size,
            shuffle=True,
            num_workers=config.dl_num_workers,
            pin_memory=True,
        )
        self.__val_loader = DataLoader(
            val_dataset,
            batch_size=self.__batch_size,
            shuffle=False,
            num_workers=config.dl_num_workers,
            pin_memory=True,
        )
        # model
        self.__model = ECGformer(
            config.model.signal_length,
            config.model.signal_channels,
            config.model.classes,
            config.model.embed_size,
            config.model.encoder_layers_num,
            config.model.encoder_heads,
            config.model.dropout,
        )
        # optimizer
        self.__optimizer = torch.optim.AdamW(  # pyright: ignore[reportPrivateImportUsage]
            self.__model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        # loss function
        self.__criterion = nn.CrossEntropyLoss()
        # device
        self.__device = torch.device(config.device)
        self.__model.to(self.__device)
        # others
        self.__epochs = config.epochs
        self.__validation_interval = config.validation_interval
        self.__save_interval = config.save_interval
        self.__start_timestamp = datetime.now().strftime("%Y%m%d%H%M")
        self.__save_dir = Path(f"./checkpoints/{self.__start_timestamp}/")
        # create save directory if not exists
        if not self.__save_dir.exists():
            self.__save_dir.mkdir(parents=True, exist_ok=True)
        # print config
        config_json = json.dumps(asdict(config), indent=4)
        print(config_json)
        print()
        # save config info to checkpoint directory
        with open(self.__save_dir.joinpath("config.json"), "w") as f:
            f.write(config_json)

    def __train_epoch(self, epoch: int) -> None:
        self.__model.train()
        loader = tqdm(self.__train_loader)
        accuracy = 0
        for data in loader:
            self.__optimizer.zero_grad()
            signal, label = [d.to(self.__device) for d in data]
            prediction = self.__model(signal.permute(0, 2, 1))
            loss = self.__criterion(prediction, label)
            loss.backward()
            self.__optimizer.step()
            accuracy += torch.sum(prediction.argmax(1) == label)
            loader.set_description(
                f"TRAINING: {epoch}, loss: {loss.item()}. Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}"
            )
        print(f"TRAINING Accuracy: {accuracy / len(loader) / self.__batch_size}")
        print()

    @torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
    def __validation_epoch(self, epoch: int) -> None:
        self.__model.eval()
        loader = tqdm(self.__val_loader)
        accuracy = 0
        for data in loader:
            signal, label = [d.to(self.__device) for d in data]
            prediction = self.__model(signal.permute(0, 2, 1))
            accuracy += torch.sum(prediction.argmax(1) == label)
            loader.set_description(
                f"VALIDATION: {epoch}, Target: {label[:8].tolist()}, Prediction: {prediction.argmax(1)[:8].tolist()}"
            )
        print(f"VALIDATION Accuracy: {accuracy / len(loader) / self.__batch_size}")
        print()

    def train(self) -> None:
        for epoch in range(1, self.__epochs + 1):
            self.__train_epoch(epoch)
            if epoch % self.__validation_interval == 0:
                self.__validation_epoch(epoch)
            if epoch % self.__save_interval == 0:
                torch.save(
                    self.__model.state_dict(),
                    f"{self.__save_dir}/model_{epoch}.pth",
                )


if __name__ == "__main__":
    config = Config()
    dataset = NPYDataset(["N", "A", "V", "L", "R"])
    trainer = Trainer(config, dataset)
    trainer.train()
