"""Bake la temporal_fill dans le cache slim (une seule fois).

Lit le cache brut (build_input_cache), et pour CHAQUE feature charge la série temporelle
CONTINUE (tous les jours concaténés), applique exactement le meme temporal_fill que
l'inférence (importé du script d'inférence), et écrit un cache "filled" sans trous +
une variable interp_count (uint8 = nombre de features interpolées par heure et par cellule)
pour reconstruire interp_fraction à l'inférence sans aucun calcul.

Le cache brut n'est jamais modifié (on écrit dans un nouveau dossier) -> rien cassé.
La temporal_fill vectorisée passe en float64 ; on la applique par bandes de latitude
pour borner la RAM (série complète résidente ~4.5 Go + bande ~1.5 Go).
"""
from __future__ import annotations
import argparse
import sys
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_output_daily_maps_assim as M  # noqa: E402  (temporal_fill, to_numpy partagés)
from build_input_cache import FEATURES  # noqa: E402

T = 24  # heures par jour


def discover(raw_dir: Path, start, end):
    days, paths = [], {}
    for f in sorted(raw_dir.glob("cache_*.nc")):
        s = f.stem.removeprefix("cache_")
        if len(s) == 8 and s.isdigit():
            d = datetime.strptime(s, "%Y%m%d").date()
            if start <= d <= end:
                days.append(d)
                paths[d] = f
    days.sort()
    return days, paths


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="output_daily_cache")
    p.add_argument("--out-dir", default="output_daily_cache_filled")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--complevel", type=int, default=1)
    p.add_argument("--band", type=int, default=48, help="lat rows per fill band (RAM bound)")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    days, paths = discover(raw, start, end)
    assert days, "aucun jour de cache brut dans l'intervalle"

    with netCDF4.Dataset(paths[days[0]]) as d0:
        lat = M.to_numpy(d0.variables["latitude"][:])
        lon = M.to_numpy(d0.variables["longitude"][:])
    H, W, N = lat.size, lon.size, len(days)
    Ntot = N * T
    print({"days": N, "grid": [H, W], "Ntot": Ntot}, flush=True)

    interp_count = np.zeros((Ntot, H, W), dtype=np.uint8)  # # features interpolées (<=23) par heure/cellule

    # Phase 1 : créer les fichiers de sortie (coords + variables vides) -- atomique par fichier (tmp->replace)
    for d in days:
        op = out / f"cache_{d.strftime('%Y%m%d')}.nc"
        if op.exists() and not args.overwrite:
            continue
        tmp = op.with_suffix(".tmp.nc")
        with netCDF4.Dataset(paths[d]) as src, netCDF4.Dataset(tmp, "w") as o:
            o.createDimension("time", T)
            o.createDimension("latitude", H)
            o.createDimension("longitude", W)
            vt = o.createVariable("time", "f8", ("time",))
            for a in src.variables["time"].ncattrs():
                vt.setncattr(a, src.variables["time"].getncattr(a))
            vt[:] = src.variables["time"][:]
            o.createVariable("latitude", "f4", ("latitude",))[:] = lat
            o.createVariable("longitude", "f4", ("longitude",))[:] = lon
            for name in FEATURES:
                o.createVariable(
                    name, "f4", ("time", "latitude", "longitude"),
                    zlib=args.complevel > 0, complevel=args.complevel,
                    chunksizes=(T, H, W), fill_value=np.float32(np.nan),
                )
            o.createVariable(
                "interp_count", "u1", ("time", "latitude", "longitude"),
                zlib=True, complevel=args.complevel, chunksizes=(T, H, W), fill_value=np.uint8(0),
            )
            o.source = "temporally-filled slim input cache; sss=sos; interp_count=#features temporally interpolated per hour"
        tmp.replace(op)

    bands = [(b, min(b + args.band, H)) for b in range(0, H, args.band)]

    # Phase 2 : remplissage par feature (série continue), bandes de latitude pour la RAM
    for fi, name in enumerate(FEATURES):
        series = np.empty((Ntot, H, W), dtype=np.float32)
        for i, d in enumerate(days):
            with netCDF4.Dataset(paths[d]) as src:
                series[i * T:(i + 1) * T] = M.to_numpy(src.variables[name][:])
        filled_frac = 0.0
        for b0, b1 in bands:
            sub = series[:, b0:b1, :]
            was_nan = np.isnan(sub)
            sub_filled = M.temporal_fill(sub)             # MEME math que l'inférence
            got = was_nan & np.isfinite(sub_filled)
            interp_count[:, b0:b1, :] += got.astype(np.uint8)
            series[:, b0:b1, :] = sub_filled
            filled_frac += float(got.sum())
        for i, d in enumerate(days):
            op = out / f"cache_{d.strftime('%Y%m%d')}.nc"
            with netCDF4.Dataset(op, "a") as o:
                o.variables[name][:] = series[i * T:(i + 1) * T]
        print({"feature": name, "filled_frac": round(filled_frac / series.size, 6),
               "done": f"{fi + 1}/{len(FEATURES)}"}, flush=True)

    # Phase 3 : écrire interp_count
    for i, d in enumerate(days):
        op = out / f"cache_{d.strftime('%Y%m%d')}.nc"
        with netCDF4.Dataset(op, "a") as o:
            o.variables["interp_count"][:] = interp_count[i * T:(i + 1) * T]
    print({"out_dir": str(out), "max_interp_count": int(interp_count.max()),
           "mean_interp_features_per_cellhour": round(float(interp_count.mean()), 4)}, flush=True)


if __name__ == "__main__":
    main()
