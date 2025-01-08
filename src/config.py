from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class TrainerConfig:
    device: str = "cpu"
    clip: float = 0.25
    lr: float = 0.001
    batch_size: int = 64
    n_epochs: int = 100

@dataclass_json
@dataclass
class ModelConfig:
    hidden_size: int = 128
    embed_dim: int = 16
    n_layers: int = 2
    dropout_p: float = 0.2
    device: str = TrainerConfig.device
