from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ulm import TransformerModel
from ulm.units.tokenizer import UnitTokenizer


class EncodedEvalDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: UnitTokenizer):
        self.units_paths = sorted(list(Path(data_dir).glob("*.npy")))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.units_paths)

    def __getitem__(self, idx: int):
        path = self.units_paths[idx]
        key = path.stem
        units_npy = np.load(path)
        encoded_ids = self.tokenizer.encode(units_npy.tolist())
        src_ids = encoded_ids[:-1].clone()
        tgt_ids = encoded_ids[1:].clone()
        return key, src_ids, tgt_ids


def collate_fn(batch: list[list[int]]) -> tuple[List[str], torch.Tensor, List[int]]:
    keys = [b[0] for b in batch]
    src_ids = torch.cat([b[1] for b in batch])
    tgt_ids = torch.cat([b[2] for b in batch])
    seqlens = [len(b[1]) for b in batch]
    return keys, src_ids, tgt_ids, seqlens


def compute_loglikelihoods(
    units_dir: str,
    checkpoint_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
):
    model = TransformerModel.from_pretrained_checkpoint(checkpoint_path).cuda()
    model.eval()

    tokenizer = UnitTokenizer()

    print(f"Computing loglikelihoods for units in {units_dir}...")

    dataset = EncodedEvalDataset(
        data_dir=units_dir,
        tokenizer=tokenizer,
    )

    print(f"Found {len(dataset)} unit files")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    data = []
    with torch.inference_mode():
        for keys, src_ids, tgt_ids, seqlens in tqdm(dataloader):
            logits = model.forward(src_ids.cuda(), seqlens=seqlens)
            neg_obs_log_probs = torch.nn.functional.cross_entropy(
                logits, tgt_ids.cuda(), reduction="none"
            )

            starts = [sum(seqlens[:i]) for i in range(len(seqlens))]
            stops = [starts[i] + seqlens[i] for i in range(len(seqlens))]

            for key, start, stop in zip(keys, starts, stops):
                ll = -neg_obs_log_probs[start:stop].sum().item()
                data.append((key, ll))

    strings = [f"{key} {ll}" for key, ll in data]
    string = "\n".join(strings)
    Path(output_path).write_text(string)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute log-likelihoods for unit sequences"
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input directory containing unit files"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output file path for log-likelihoods (.txt)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for processing (default: 128)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=16,
        help="Number of workers for processing (default: 16)",
    )

    args = parser.parse_args()

    compute_loglikelihoods(
        units_dir=args.input,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
