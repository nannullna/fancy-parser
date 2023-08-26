import gradio as gr

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
    # optimizer: str = field(default='sgd', metadata={'help': 'Optimizer to use.', 'choices': ['sgd', 'adam']})
    save_every: Optional[int] = field(default=None, metadata={'help': 'Save model every n epochs.'})
    use_wandb: bool = field(default=False, metadata={'help': 'Use wandb for logging.'})

@dataclass
class ModelArgs:
    """Model arguments."""
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64], metadata={'help': 'Hidden layer sizes.'})
    dropout: float = field(default=0.1, metadata={'help': 'Dropout rate.'})
    activation: str = field(default='relu', metadata={'help': 'Activation function.'})


def build_interface_from_dataclass(cls):
    """Build a gradio interface from a dataclass for hyperparameters."""
    def _build_interface_from_dataclass(cls):
        """Build a gradio interface from a dataclass for hyperparameters."""
        fields = cls.__dataclass_fields__
        inputs = []
        for name, field in fields.items():
            if field.type == bool:
                inputs.append(gr.inputs.Checkbox(label=name))
            elif field.type == int:
                inputs.append(gr.inputs.Number(label=name))
            elif field.type == float:
                inputs.append(gr.inputs.Slider(label=name, minimum=0.0, maximum=1.0, step=0.01))
            elif field.type == str:
                inputs.append(gr.inputs.Textbox(label=name))
            elif field.type == List[int]:
                inputs.append(gr.inputs.Number(label=name))
            elif field.type == List[float]:
                inputs.append(gr.inputs.Slider(label=name, minimum=0.0, maximum=1.0, step=0.01))
            elif field.type == List[str]:
                inputs.append(gr.inputs.Textbox(label=name))
            else:
                pass
                # raise NotImplementedError(f"Type {field.type} not supported.")
        return inputs
    return gr.Interface(fn=lambda **kwargs: print(kwargs), inputs=_build_interface_from_dataclass(cls), outputs=None)


classes = [TrainingArgs, ModelArgs]
interfaces = [build_interface_from_dataclass(cls) for cls in classes]
demo = gr.TabbedInterface(interfaces, [cls.__name__ for cls in classes])

if __name__ == '__main__':
    demo.launch(share=True)
