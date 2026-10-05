from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morlenn.config import load_config
from morlenn.data import MonthlyNetCDFDataset, list_netcdf_files, load_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute UpperDyn NetCDF samples to .pt shards.")
    parser.add_argument("--config", required=True, help="Path to config TOML.")
    parser.add_argument("--split", choices=("train", "val"), default=None,
                        help="Legacy per-split mode (reads config train_dir/val_dir + *_precomputed_dir).")
    parser.add_argument("--input-dir", default=None,
                        help="Global mode: directory of NetCDF months to precompute (e.g. data/mld_split/all).")
    parser.add_argument("--out-dir", default=None,
                        help="Global mode: output precompute directory (e.g. data/pool/precompute/all).")
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.input_dir is not None or args.out_dir is not None:
        # Global mode: one precompute over all months, split deferred to training.
        if args.input_dir is None or args.out_dir is None:
            raise ValueError("Global mode requires BOTH --input-dir and --out-dir.")
        if args.split is not None:
            raise ValueError("--split is incompatible with --input-dir/--out-dir (global mode).")
        label = "all"
        directory = Path(args.input_dir)
        output_dir = Path(args.out_dir)
    else:
        if args.split is None:
            raise ValueError("Provide either --split {train,val} (legacy) or --input-dir/--out-dir (global).")
        label = args.split
        directory = config.paths.train_dir if args.split == "train" else config.paths.val_dir
        output_dir = (
            config.dataset.train_precomputed_dir
            if args.split == "train"
            else config.dataset.val_precomputed_dir
        )
        if output_dir is None:
            raise ValueError(f"Missing {args.split}_precomputed_dir in config.")
        output_dir = Path(output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("shard_*.pt")) or (output_dir / "dataset.pt").exists():
        raise FileExistsError(f"Precomputed shards already exist in {output_dir}. Use --overwrite.")

    stats = load_stats(config.paths.stats_path)
    dataset = MonthlyNetCDFDataset(
        files=list_netcdf_files(directory),
        dynamic_features=config.dataset.dynamic_features,
        static_features=config.dataset.static_features,
        targets=config.dataset.targets,
        mld_target_transform=config.dataset.mld_target_transform,
        stats=stats,
        replace_nan_with_zero=config.dataset.replace_nan_with_zero,
        clip_mld_max=config.dataset.clip_mld_max,
        dynamic_history_hours=config.dataset.dynamic_history_hours,
        strict_static_features=config.dataset.strict_static_features,
        eddy_max_distance_radius=config.dataset.eddy_max_distance_radius,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.shard_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    metadata = {
        "split": label,
        "num_samples": len(dataset),
        "dynamic_features": config.dataset.dynamic_features,
        "static_features": config.dataset.static_features,
        "strict_static_features": config.dataset.strict_static_features,
        "targets": config.dataset.targets,
        "mld_target_transform": config.dataset.mld_target_transform,
        "stats_path": str(config.paths.stats_path),
        "shard_size": args.shard_size,
        "single_file": args.single_file,
    }
    torch.save(metadata, output_dir / "metadata.pt")

    if args.single_file:
        chunks: dict[str, list[torch.Tensor]] = {"dynamic": [], "static": [], "target": []}
        for shard_idx, batch in enumerate(tqdm(loader, desc=f"precompute {label}", dynamic_ncols=True)):
            payload = {
                "dynamic": batch["dynamic"].contiguous().to(torch.float32),
                "static": batch["static"].contiguous().to(torch.float32),
                "target": batch["target"].contiguous().to(torch.float32),
            }
            for name, tensor in payload.items():
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite {name} tensor while precomputing {label} shard {shard_idx}.")
                chunks[name].append(tensor)
        torch.save({name: torch.cat(parts, dim=0) for name, parts in chunks.items()}, output_dir / "dataset.pt")
        print(f"Saved {len(dataset)} samples to {output_dir / 'dataset.pt'}")
        return

    for shard_idx, batch in enumerate(tqdm(loader, desc=f"precompute {label}", dynamic_ncols=True)):
        payload = {
            "dynamic": batch["dynamic"].contiguous().to(torch.float32),
            "static": batch["static"].contiguous().to(torch.float32),
            "target": batch["target"].contiguous().to(torch.float32),
        }
        for name, tensor in payload.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Non-finite {name} tensor while precomputing {label} shard {shard_idx}.")
        torch.save(payload, output_dir / f"shard_{shard_idx:05d}.pt")
    print(f"Saved {len(dataset)} samples to {output_dir}")


if __name__ == "__main__":
    main()
