import os

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def cli_main():
    cli = LightningCLI(
        trainer_defaults={
            'accelerator': 'gpu',
            'strategy': 'ddp',
            'log_every_n_steps': 1,
            'callbacks': [
                ModelCheckpoint(
                    save_top_k=1,
                    save_last=True,
                    every_n_train_steps=10000,
                    filename='{epoch}-{step}',
                ),
                ModelSummary(max_depth=4)
            ]
        }
    )


if __name__ == "__main__":
    cli_main()
