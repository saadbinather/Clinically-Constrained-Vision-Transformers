from dataclasses import dataclass, asdict
from pathlib import Path
import json

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0  # Reliable default on macOS; override in config if desired.
EPOCHS = 10

@dataclass(frozen=True)
class RunConfig:
    name: str
    dataset: str
    architecture: str
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    seed: int = SEED
    lr: float = 1e-4
    weight_decay: float = 0.0
    pretrained: bool = True
    alignment: bool = False
    lambda_weight: float = 0.0

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

CONFIGS = [
    RunConfig("nih_densenet121", "nih", "densenet121"),
    RunConfig("nih_resnet50", "nih", "resnet50"),
    RunConfig("nih_vit_base", "nih", "vit_base_patch16_224", lr=1e-5, weight_decay=1e-2),
    RunConfig("chexpert_densenet121", "chexpert", "densenet121"),
    RunConfig("chexpert_resnet50", "chexpert", "resnet50"),
    RunConfig("chexpert_vit_base", "chexpert", "vit_base_patch16_224", lr=1e-5, weight_decay=1e-2),
    RunConfig("rsna_vit_no_cacr", "rsna", "vit_base_patch16_224", lr=1e-5, alignment=True, lambda_weight=0.0),
    RunConfig("rsna_vit_cacr", "rsna", "vit_base_patch16_224", lr=1e-5, alignment=True, lambda_weight=1.0),
]
