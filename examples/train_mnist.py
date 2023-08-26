from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Literal
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
import torchvision.transforms as T

from fancy_parser import FancyParser


@dataclass
class ModelArguments:
    num_classes: int = 10
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    dropout: float = 0.1
    activation: str = field(default='relu', metadata={'choices': ['relu', 'tanh']})

@dataclass
class TrainingArguments:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    optimizer: str = field(default='adam', metadata={'choices': ['sgd', 'adam', 'adamw']})
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_dir: str = 'mnist'

def get_activation(activation: str) -> nn.Module:
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError(f'Invalid activation {activation}')

def create_model(model_args: ModelArguments) -> nn.Module:
    layers = []
    prev_dim = 28 * 28
    for idx, dim in enumerate(model_args.hidden_dims):
        layer = nn.Sequential()
        layer.add_module('linear', nn.Linear(prev_dim, dim))
        layer.add_module('activation', get_activation(model_args.activation))
        layer.add_module('dropout', nn.Dropout(model_args.dropout))
        layers.append(layer)
        prev_dim = dim
    layers = nn.Sequential(*layers)
    layers.add_module('classifier', nn.Linear(prev_dim, model_args.num_classes))
    return layers

def create_optimizer(model: nn.Module, training_args: TrainingArguments) -> optim.Optimizer:
    if training_args.optimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    elif training_args.optimizer == 'adam':
        return optim.Adam(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    elif training_args.optimizer == 'adamw':
        return optim.AdamW(model.parameters(), lr=training_args.lr, weight_decay=training_args.weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {training_args.optimizer}')

def train_step(model: nn.Module, optimizer: optim.Optimizer, device: str, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    x, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean()
    return {'loss': loss.item(), 'acc': acc.item(), 'n': len(x)}

@torch.no_grad()
def eval_step(model: nn.Module, device: str, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    x, y = batch[0].to(device), batch[1].to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    preds = torch.argmax(logits, dim=-1)
    acc = (preds == y).float().mean()
    return {'loss': loss.item(), 'acc': acc.item(), 'n': len(x)}

def train_epoch(model: nn.Module, optimizer: optim.Optimizer, device: str, train_loader: DataLoader) -> Dict[str, Any]:
    logs = []
    for batch in train_loader:
        logs.append(train_step(model, optimizer, device, batch))
    logs = {k: np.average([log[k] for log in logs], weights=[log['n'] for log in logs]) for k in logs[0]}
    return logs

@torch.no_grad()
def eval_epoch(model: nn.Module, device: str, val_loader: DataLoader) -> Dict[str, Any]:
    logs = []
    for batch in val_loader:
        logs.append(eval_step(model, device, batch))
    logs = {k: np.average([log[k] for log in logs], weights=[log['n'] for log in logs]) for k in logs[0]}
    return logs

def train(train_args: TrainingArguments, model: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, val_loader: DataLoader) -> None:
    model.to(train_args.device)
    optimizer = create_optimizer(model, train_args)
    for epoch in range(train_args.epochs):
        model.train()
        train_logs = train_epoch(model, optimizer, train_args.device, train_loader)
        model.eval()
        valid_logs = []
        for batch in val_loader:
            valid_logs.append(eval_step(model, train_args.device, batch))
        valid_logs = {k: np.average([log[k] for log in valid_logs], weights=[log['n'] for log in valid_logs]) for k in valid_logs[0]}
        print(f'  Epoch {epoch+1} - loss: {train_logs["loss"]:.4f} - acc: {train_logs["acc"]:.4f} - val_loss: {valid_logs["loss"]:.4f} - val_acc: {valid_logs["acc"]:.4f}')

def main():
    parser = FancyParser([TrainingArguments, ModelArguments], description='Train a simple MLP on MNIST')
    train_args, model_args = parser.parse_args_into_dataclasses()
    train_args: TrainingArguments
    model_args: ModelArguments
    print(train_args)
    print(model_args)

    if not os.path.exists(train_args.dataset_dir):
        os.makedirs(train_args.dataset_dir)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        # flatten
        T.Lambda(lambda x: x.view(-1)),
    ])
    
    train_dataset = MNIST(train_args.dataset_dir, train=True, download=True, transform=transform)
    valid_dataset = MNIST(train_args.dataset_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=train_args.batch_size, shuffle=False, num_workers=4)

    model = create_model(model_args)
    optimizer = create_optimizer(model, train_args)

    train(train_args, model, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    main()
