#!/usr/bin/env python3
"""Evaluate SWA checkpoint (raw argmax) on val_holdout.

Usage:
    python scripts/eval_swa_holdout.py --run-dir output/runs/<run_name>
    python scripts/eval_swa_holdout.py --run-dir output/runs/<run_name> --decode expectation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common.data.grouped_dataset import GroupedParticipantDataset, grouped_collate_fn
from common.data.hdf5_dataset import HDF5GroupedDataset
from common.runner import _decode_a2_logits, _normalize_decode_method, _to_device
from common.utils.metrics import mean_qwk, mean_mae
from scripts.run_predict_a2 import build_model_from_config

log = logging.getLogger("eval_swa")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/data1/AdoDas"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--decode", default="argmax", choices=["argmax", "expectation", "monotonic"])
    p.add_argument("--checkpoint", default="swa.pt")
    return p.parse_args()


def load_swa_model(cfg: dict, ckpt_path: Path, device: torch.device):
    """Reconstruct model from config, load SWA checkpoint."""
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Extract state dicts (handle both MTL and non-MTL checkpoints)
    full_sd = ckpt["model_state_dict"]
    has_mtl_prefix = any(k.startswith("grouped_model.") for k in full_sd)
    if has_mtl_prefix:
        model_sd = {}
        head_sd = {}
        for k, v in full_sd.items():
            if k.startswith("grouped_model."):
                model_sd[k[len("grouped_model."):]] = v
            elif k.startswith("participant_head."):
                head_sd[k[len("participant_head."):]] = v
    elif "participant_head_state_dict" in ckpt:
        model_sd = {k: v for k, v in full_sd.items()}
        head_sd = ckpt["participant_head_state_dict"]
    else:
        model_sd = {k: v for k, v in full_sd.items()}
        head_sd = {}

    grouped_model, participant_head, feat_cfg, _, _ = build_model_from_config(cfg, model_sd)
    grouped_model.load_state_dict(model_sd, strict=False)
    participant_head.load_state_dict(head_sd, strict=True)
    grouped_model.to(device).eval()
    participant_head.to(device).eval()
    return grouped_model, participant_head, feat_cfg


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    args = parse_args()

    meta_path = args.run_dir / "run_meta.json"
    ckpt_path = args.run_dir / "checkpoints" / args.checkpoint
    if not meta_path.exists() or not ckpt_path.exists():
        log.error("Missing run_meta.json or %s checkpoint", args.checkpoint)
        return 1

    meta = json.load(open(meta_path))
    cfg = meta.get("full_config") or {}
    cfg_path = args.run_dir / "config_used.yaml"
    if not cfg and cfg_path.exists():
        cfg = yaml.safe_load(open(cfg_path)) or {}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Build model & load SWA
    grouped_model, participant_head, feat_cfg = load_swa_model(cfg, ckpt_path, device)

    # Auto-match SSL tags if HDF5 features differ from config
    from scripts.run_predict_a2 import _match_ssl_tag
    pass  # feat_cfg is already built with config values; HDF5 should match

    # Load val_split to get holdout PIDs
    split_path = Path("splits/val_split_v1.json")
    val_holdout_pids = None
    if split_path.exists():
        split = json.load(open(split_path))
        val_holdout_pids = set(split.get("val_holdout_pids", []))
        log.info("val_holdout PIDs: %d", len(val_holdout_pids))

    # Build val_holdout dataset
    val_csv = args.data_root / "Val" / "val.csv"
    hdf5_path = args.data_root / "val_packed.h5"
    if hdf5_path.exists():
        ds = HDF5GroupedDataset(str(hdf5_path), valid_pids=val_holdout_pids)
        log.info("Using HDF5 val dataset")
    else:
        ds = GroupedParticipantDataset(val_csv, cfg, split="val", valid_pids=val_holdout_pids)
        log.info("Using raw val dataset")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=grouped_collate_fn)

    # Inference
    decode_method = _normalize_decode_method(args.decode)
    all_preds, all_labels = [], []
    for batch in loader:
        if batch is None:
            continue
        flat = _to_device(batch["flat_batch"], device)
        valid = batch["session_valid"].to(device)
        aux = batch.get("participant_aux_attrs")
        aux = aux.to(device) if aux is not None else None
        B = batch["n_participants"]

        with torch.no_grad():
            out = grouped_model(flat, B, valid, aux)
            logits = participant_head(out["participant_repr"]).float()
        preds = _decode_a2_logits(participant_head, logits, decode_method=decode_method)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(batch["participant_y_a2"].numpy())

    preds = np.concatenate(all_preds).astype(int)
    labels = np.concatenate(all_labels).astype(int)

    qwk = mean_qwk(preds, labels)
    mae = mean_mae(preds, labels)

    log.info("=" * 60)
    log.info("SWA raw %s on val_holdout (%d samples):", args.decode, len(preds))
    log.info("  QWK = %.4f", qwk)
    log.info("  MAE = %.4f", mae)
    total = preds.size
    dist = [np.sum(preds == v) / total * 100 for v in range(4)]
    log.info("  dist: 0=%.1f%% 1=%.1f%% 2=%.1f%% 3=%.1f%%", *dist)
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
