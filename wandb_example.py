from src.fancy_parser import FancyParser
from dataclasses import dataclass, field
from typing import Optional, Union, List
from enum import Enum

import wandb

class Optimizer(Enum):
    SGD = 'sgd'
    ADAM = 'adam'


@dataclass
class TrainingArgs:
    """Training arguments."""
    epochs: int = field(default=10, metadata={'help': 'Number of epochs to train for.'})
    batch_size: int = field(default=32, metadata={'help': 'Batch size for training.'})
    optimizer: Optimizer = field(default=Optimizer.SGD, metadata={'help': 'Optimizer to use.'})
    # optimizer: str = field(default='sgd', metadata={'help': 'Optimizer to use.', 'choices': ['sgd', 'adam']})
    save_every: Optional[int] = field(default=None, metadata={'help': 'Save model every n epochs.'})
    use_wandb: bool = field(default=False, metadata={'help': 'Use wandb for logging.'})


@dataclass
class ModelArgs:
    """Model arguments."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64], metadata={'help': 'Hidden layer sizes.'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate.'})
    activation: str = field(default='relu', metadata={'help': 'Activation function.'})


def init_model(model_args: ModelArgs, **kwargs):
    print(f"Constructing model with {len(model_args.hidden_dims)} hidden layers.")
    prev_dim = 28 * 28
    for dim in model_args.hidden_dims:
        print(f"  Adding layer with {prev_dim} inputs and {dim} outputs")
        print(f"  + activation {model_args.activation}.")
        print(f"  + dropout {model_args.dropout}.")
        prev_dim = dim
    print(f"  Adding output layer with {prev_dim} inputs and 10 outputs.")


def init_optimizer(training_args: TrainingArgs, **kwargs):
    print(f"Initializing optimizer {training_args.optimizer}.")


def train(training_args: TrainingArgs, **kwargs):
    print(f"Training for {training_args.epochs} epochs.")
    for epoch in range(1, training_args.epochs+1):
        print(f"  Epoch {epoch} / {training_args.epochs}")
        if training_args.save_every and (epoch % training_args.save_every == 0):
            print("    Saving model...")
            # save_model()


def main():
    parser = FancyParser([TrainingArgs, ModelArgs])
    # We here parse the arguments into a namespace.
    config = parser.parse_args()
    if config.use_wandb:
        # Wandb reports hyperparameters as a dict 
        # and may change some values for sweep.
        wandb.init(project="example", config=config)
        config = wandb.config
    
    # Now we can parse the config dict given by wandb.
    training_args, model_args = parser.parse_dict(vars(config), allow_extra_keys=True)
    print(training_args)
    print(model_args)

    init_model(model_args)
    init_optimizer(training_args)
    train(training_args)

    wandb.finish()
    

if __name__ == '__main__':
    main()