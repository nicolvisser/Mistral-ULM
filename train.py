from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from simple_parsing import Serializable
from torch.utils.data import DataLoader

import wandb
from ulm import (
    LinearRampCosineDecayScheduler,
    TransformerLightningModule,
    TransformerModel,
    TransformerModelArgs,
    collate_fn,
)
from ulm.units import (
    TokenizedUnitsChunkedDataset,
    TokenizedUnitsUtteranceDataset,
    UnitTokenizer,
)

# DEFINE TRAIN ARGS


@dataclass
class TrainArgs(Serializable):
    """Training arguments for the transformer model."""

    project_name: str
    run_name: str
    train_dataset_dir: str
    train_dataset_pattern: str
    valid_dataset_dir: str
    valid_dataset_pattern: str
    chunk_size: int
    batch_size: int
    num_workers: int
    lr_init: float
    warmup_steps: int
    lr_max: float
    decay_steps: int
    lr_final: float
    betas: tuple[float, float]
    weight_decay: float
    eps: float
    accelerator: str
    strategy: str
    devices: int
    precision: str
    fast_dev_run: bool
    max_steps: int
    val_check_interval: float
    check_val_every_n_epoch: int
    log_every_n_steps: int
    accumulate_grad_batches: int
    gradient_clip_algorithm: str
    gradient_clip_val: float


def train(
    model_config_path: str = "model_config.json",
    train_config_path: str = "train_config.json",
) -> None:
    """Train the transformer model.

    Args:
        model_config_path: Path to the model configuration file
        train_config_path: Path to the training configuration file
    """
    # LOAD ALL ARGS

    model_args = TransformerModelArgs.load_json(model_config_path)
    train_args = TrainArgs.load_json(train_config_path)

    # DATASET SETUP

    tokenizer = UnitTokenizer()

    train_dataset = TokenizedUnitsChunkedDataset(
        units_dir=train_args.train_dataset_dir,
        pattern=train_args.train_dataset_pattern,
        tokenizer=tokenizer,
        max_chunk_size=train_args.chunk_size,
    )
    valid_dataset = TokenizedUnitsUtteranceDataset(
        units_dir=train_args.valid_dataset_dir,
        pattern=train_args.valid_dataset_pattern,
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_args.batch_size,
        shuffle=True,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=train_args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # CHECKPOINT SETUP

    checkpoint_dir = Path("./checkpoints") / train_args.run_name
    model_args_path = checkpoint_dir / "model_args.json"
    train_args_path = checkpoint_dir / "train_args.json"
    lighting_best_checkpoint_path = checkpoint_dir / "best.ckpt"
    torch_best_checkpoint_path = checkpoint_dir / "best.pt"

    if checkpoint_dir.exists():
        msg = f"Warning: Checkpoint directory {checkpoint_dir} already exists. Continue and overwrite contents? (y/n): "
        response = input(msg)
        if response.lower() != "y":
            print("Training aborted.")
            exit()

    class CustomCheckpointCallback(ModelCheckpoint):
        """Custom checkpoint callback to save model args and vanilla torch checkpoint."""

        def on_validation_end(
            self, trainer, pl_module: TransformerLightningModule
        ) -> None:
            super().on_validation_end(trainer, pl_module)
            # save the model args and trainer args as json files
            if not checkpoint_dir.exists():
                checkpoint_dir.mkdir(parents=True)
            model_args.save_json(model_args_path, indent=4)
            train_args.save_json(train_args_path, indent=4)
            # save the vanilla torch model as well
            model: TransformerModel = pl_module.model
            model.save_pretrained_checkpoint(torch_best_checkpoint_path)

    checkpoint_callback = CustomCheckpointCallback(
        dirpath=checkpoint_dir,
        filename=lighting_best_checkpoint_path.stem,
        monitor="val/loss",
        verbose=True,
        save_last=True,
        save_top_k=1,
        save_weights_only=True,
        mode="min",
    )

    # LOGGER SETUP

    logger = pl.loggers.WandbLogger(
        log_model=False,
        project=train_args.project_name,
        name=train_args.run_name,
    )

    # Log hyperparameters
    logger.log_hyperparams(
        {
            "model_args": model_args.to_dict(),
            "train_args": train_args.to_dict(),
        }
    )

    # LR MONITOR SETUP

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")

    # MODEL SETUP

    model = TransformerLightningModule(model_args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_args.lr_max,
        betas=train_args.betas,
        weight_decay=train_args.weight_decay,
        eps=train_args.eps,
    )

    scheduler = {
        "scheduler": LinearRampCosineDecayScheduler(
            optimizer,
            n_linear_steps=train_args.warmup_steps,
            n_decay_steps=train_args.decay_steps,
            lr_init=train_args.lr_init,
            lr_max=train_args.lr_max,
            lr_final=train_args.lr_final,
        ),
        "frequency": 1,
        "interval": "step",
    }

    model.configure_optimizers = lambda: {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
    }

    # TRAINER SETUP

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            checkpoint_callback,
            lr_monitor_callback,
        ],
        accelerator=train_args.accelerator,
        strategy=train_args.strategy,
        devices=train_args.devices,
        precision=train_args.precision,
        fast_dev_run=train_args.fast_dev_run,
        max_steps=train_args.max_steps,
        val_check_interval=train_args.val_check_interval,
        check_val_every_n_epoch=train_args.check_val_every_n_epoch,
        log_every_n_steps=train_args.log_every_n_steps,
        accumulate_grad_batches=train_args.accumulate_grad_batches,
        gradient_clip_algorithm=train_args.gradient_clip_algorithm,
        gradient_clip_val=train_args.gradient_clip_val,
    )

    # SAVE CURRENT SCRIPT

    wandb.save(str(Path(__file__).resolve()))

    # TRAINING

    torch.set_float32_matmul_precision("medium")  # Optimize for some NVIDIA GPUs

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a transformer model")
    parser.add_argument(
        "-m",
        "--model-config",
        type=str,
        default="model_config.json",
        help="Path to the model configuration file",
    )
    parser.add_argument(
        "-t",
        "--train-config",
        type=str,
        default="train_config.json",
        help="Path to the training configuration file",
    )

    args = parser.parse_args()
    train(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )
