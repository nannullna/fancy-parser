from src.fancy_parser import FancyParser
from dataclasses import dataclass, field
from typing import Optional, Union, List
from enum import Enum


class Optimizer(Enum):
    SGD = 'sgd'
    ADAM = 'adam'


@dataclass
class TrainingArgs:
    """Training arguments."""
    epochs: int = field(default=10, metadata={'help': 'Number of epochs to train for.'})
    batch_size: int = field(default=32, metadata={'help': 'Batch size for training.'})
    optimizer: Optimizer = field(default=Optimizer.SGD, metadata={'help': 'Optimizer to use.'})
    save_every: Optional[int] = field(default=None, metadata={'help': 'Save model every n epochs.'})


@dataclass
class ModelArgs:
    """Model arguments."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64], metadata={'help': 'Hidden layer sizes.'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate.'})
    activation: str = field(default='relu', metadata={'help': 'Activation function.'})


if __name__ == '__main__':
    parser = FancyParser([TrainingArgs, ModelArgs])
    training_args, model_args = parser.parse_args_into_dataclasses()
    print(training_args)
    print(model_args)

