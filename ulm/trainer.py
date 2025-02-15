from pathlib import Path

import lightning as L
import torch

from .data import TokenizedBatch
from .transformer import TransformerModel, TransformerModelArgs


class TransformerLightningModule(L.LightningModule):
    def __init__(
        self,
        model_args: TransformerModelArgs,
    ):
        super().__init__()
        self.model = TransformerModel(args=model_args)

    def forward(self, batch: TokenizedBatch) -> torch.Tensor:
        logits = self.model.forward(input_ids=batch.src_ids, seqlens=batch.seqlens)
        logits_for_loss = logits[batch.loss_mask]
        tgt_ids_for_loss = batch.tgt_ids[batch.loss_mask]
        loss = torch.nn.functional.cross_entropy(
            input=logits_for_loss,
            target=tgt_ids_for_loss,
        )
        return loss

    def training_step(self, batch: TokenizedBatch, batch_idx: int) -> torch.Tensor:
        loss = self(batch=batch)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: TokenizedBatch, batch_idx: int) -> None:
        loss = self(batch=batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            batch_size=batch.batch_size,
            sync_dist=True,
        )

    def configure_optimizers(self):
        return None  # optimizers will be configured externally

    @classmethod
    def from_model_dir(
        cls,
        model_dir: str,
        checkpoint_filename: str = "best.ckpt",
        model_args_filename: str = "model_args.json",
    ):
        model_dir = Path(model_dir)
        checkpoint_path: Path = model_dir / checkpoint_filename
        model_args_path: Path = model_dir / model_args_filename
        assert checkpoint_path.exists(), checkpoint_path
        assert model_args_path.exists(), model_args_path
        return cls.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model_args=TransformerModelArgs.load(model_args_path),
            strict=False,
        )
