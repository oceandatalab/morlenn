#!/usr/bin/env python
"""eval_checkpoint.py — évalue un ou plusieurs checkpoints sur le set de validation et sort
le TABLEAU COMPLET des métriques MLD/N2 (pour choisir le meilleur checkpoint).

Utile surtout pour le mode PROD (val_years=[]) où aucune validation n'est loggée pendant
l'entraînement : on choisit le checkpoint final a posteriori.

  python eval_checkpoint.py --config prod/config/profiles/assim.toml --val-years 2023 --gpu 0 \
      prod/train/assim/checkpoints/<run>/              # -> tous les step_*.pt du run
  python eval_checkpoint.py --config .../assim.toml a.pt b.pt --gpu 0   # checkpoints explicites

Le loader de validation est construit UNE seule fois puis réutilisé pour chaque checkpoint.
"""
from __future__ import annotations
import argparse, os, sys, glob
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]            # prod/code
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))             # pour build_model de l'infer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="checkpoint(s) .pt (globs OK) ou un dossier de run")
    ap.add_argument("--config", required=True, help="profil TOML (architecture modèle + données)")
    ap.add_argument("--val-years", default=None, help="ex '2023' ou '2022,2023' ; défaut = val_years du config")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--sort", default="mld_tail_calibrated", help="métrique de tri (croissant)")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    import torch
    from morlenn.config import load_config
    from morlenn.data import load_stats
    from morlenn.losses import WeightedUpperDynLoss
    from morlenn.train import run_epoch, build_dataloaders, selection_score_from_metrics
    from infer_output_daily_maps_stream import build_model

    # --- résoudre la liste de checkpoints (fichiers, globs, ou dossier de run) ---
    ckpts: list[str] = []
    for c in a.checkpoints:
        p = Path(c)
        if p.is_dir():
            ckpts += sorted(str(x) for x in p.glob("step_*.pt"))
        else:
            g = sorted(glob.glob(c))
            ckpts += g if g else [c]
    ckpts = list(dict.fromkeys(ckpts))
    ckpts = [c for c in ckpts if Path(c).exists()]
    if not ckpts:
        sys.exit("Aucun checkpoint trouvé.")

    cfg = load_config(a.config)
    if a.val_years is not None:
        cfg.dataset.val_years = [int(y) for y in a.val_years.split(",") if y.strip()]
    if a.batch_size:
        cfg.training.batch_size = a.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[eval] config={a.config} val_years={cfg.dataset.val_years} device={device}", flush=True)
    _train, val_loader = build_dataloaders(cfg, distributed=False)
    del _train
    n_val = getattr(val_loader, "num_samples", None)
    if not n_val:
        sys.exit(f"ERREUR: set de validation VIDE (val_years={cfg.dataset.val_years} ne sélectionne rien). "
                 f"Passe --val-years (ex --val-years 2023).")
    print(f"[eval] {n_val} échantillons de validation ; {len(ckpts)} checkpoint(s)", flush=True)

    stats = load_stats(cfg.paths.stats_path)
    L, M = cfg.loss, cfg.model
    criterion = WeightedUpperDynLoss(
        target_median=stats.target_median.tolist(), target_iqr=stats.target_iqr.tolist(),
        mld_log_weight=L.mld_log_weight, n2_task_weight=L.n2_task_weight,
        mld_deep_thresholds=L.mld_deep_thresholds, mld_deep_weights=L.mld_deep_weights,
        mld_underprediction_weight=L.mld_underprediction_weight, mld_physical_weight=L.mld_physical_weight,
        mld_physical_delta=L.mld_physical_delta, mld_rmse_weight=L.mld_rmse_weight,
        mld_tail_rmse_weight=L.mld_tail_rmse_weight, mld_spread_weight=L.mld_spread_weight,
        mld_corr_weight=L.mld_corr_weight, n2_rmse_weight=L.n2_rmse_weight, n2_spread_weight=L.n2_spread_weight,
        n2_corr_weight=L.n2_corr_weight, n2_warmup_epochs=L.n2_warmup_epochs, n2_warmup_start_factor=L.n2_warmup_start_factor,
        mld_tail_loss_quantile=L.mld_tail_loss_quantile, huber_delta=L.huber_delta, uncertainty_weight=L.uncertainty_weight,
        predictive_distribution=M.predictive_distribution, min_std=M.min_std,
        target_names=cfg.dataset.targets, mld_target_transform=cfg.dataset.mld_target_transform,
        mld_regime_thresholds=L.mld_regime_thresholds, regime_gate_weight=L.regime_gate_weight,
        regime_load_balance_weight=L.regime_load_balance_weight,
    ).to(device)
    tmed = torch.as_tensor(stats.target_median, dtype=torch.float32)
    tiqr = torch.as_tensor(stats.target_iqr, dtype=torch.float32)
    smed = torch.as_tensor(stats.static_median, dtype=torch.float32)
    siqr = torch.as_tensor(stats.static_iqr, dtype=torch.float32)
    no_neigh = not cfg.model.use_mld_neighbors

    rows = []
    for ck in ckpts:
        ckobj = torch.load(ck, map_location=device)   # build_model attend le checkpoint chargé (dict)
        model = build_model(cfg, ckobj, no_neigh, device)
        with torch.no_grad():
            m = run_epoch(
                model=model, loader=val_loader, criterion=criterion, device=device, optimizer=None,
                grad_clip=cfg.training.grad_clip, target_median=tmed, target_iqr=tiqr,
                static_median=smed, static_iqr=siqr, target_names=cfg.dataset.targets,
                static_features=cfg.dataset.static_features, mld_target_transform=cfg.dataset.mld_target_transform,
                tail_quantile=cfg.training.tail_quantile, progress=False, metric_batches=None,
                clamp_mld_to_bathy=True, regime_split=cfg.model.mld_regime_split,
            )
        vm = {f"val_{k}": v for k, v in m.items()}
        bw, sw, tw = cfg.training.checkpoint_bias_weight, cfg.training.checkpoint_std_weight, cfg.training.checkpoint_tail_weight
        m["mld_calibrated"] = selection_score_from_metrics(vm, "mld_calibrated", bw, sw, tw)[1]
        m["mld_tail_calibrated"] = selection_score_from_metrics(vm, "mld_tail_calibrated", bw, sw, tw)[1]
        m["_name"] = Path(ck).name
        rows.append(m)
        print(f"  ok {Path(ck).name}: mld_rmse={m['mld_rmse']:.4f} tail={m['mld_tail_rmse_phys']:.1f} calib={m['mld_calibrated']:.2f}", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cols = [("checkpoint", "_name", "s", 24),
            ("mld_rmse", "mld_rmse", ".4f", 9), ("mld_rmse_m", "mld_rmse_phys", ".1f", 11), ("mld_corr", "mld_corr", ".4f", 9),
            ("mld_bias_m", "mld_bias_phys", "+.1f", 11), ("mld_stdR", "mld_std_ratio", ".2f", 9),
            ("tail_rmse_m", "mld_tail_rmse_phys", ".1f", 12), ("tail_bias_m", "mld_tail_bias_phys", "+.1f", 12),
            ("n2_rmse", "n2_rmse", ".4f", 8), ("n2_corr", "n2_corr", ".4f", 8),
            ("mld_calib", "mld_calibrated", ".2f", 10), ("tail_calib", "mld_tail_calibrated", ".2f", 11), ("loss", "loss", ".1f", 8)]
    if any(a.sort == c[1] for c in cols):
        rows.sort(key=lambda r: r.get(a.sort, float("inf")))

    print("\n" + "".join(f"{name:>{w}}" for name, _, _, w in cols))
    for r in rows:
        line = ""
        for name, key, fmt, w in cols:
            v = r.get(key)
            if fmt == "s":
                cell = str(v)
            elif isinstance(v, (int, float)):
                cell = format(v, fmt)
            else:
                cell = "-"
            line += cell.rjust(w)
        print(line)

    def best(key, mode="min"):
        vals = [(r["_name"], r.get(key)) for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        n, v = (min if mode == "min" else max)(vals, key=lambda x: x[1])
        return f"{n} ({v:.3f})"
    print(f"\nMeilleurs -> mld_rmse: {best('mld_rmse')} | tail_rmse: {best('mld_tail_rmse_phys')} | "
          f"mld_corr: {best('mld_corr','max')} | mld_calib: {best('mld_calibrated')} | tail_calib: {best('mld_tail_calibrated')}")


if __name__ == "__main__":
    main()
