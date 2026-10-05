""" Precompute a slim cache of dynamical entries (I/O once)

Each file is read, necessary features are stored and additional variables
 (gradient, slope, norm) are precomputed
dynamiques exactement comme l'inférence (wind_speed_neutral, sea_slope, grad_analysed_sst
et grad_sss recalculés, sss := sos partout), et on écrit cache_YYYYMMDD.nc ne contenant
que ces 23 variables (24, lat, lon) en float32. L'inférence lit alors chaque feature
directement, sans dérivation ni recompute de gradient, et un buffer mémoire jour->feature
évite de re-décompresser le même jour pour chaque heure/tuile de sortie.

Les helpers de dérivation sont importés du script d'inférence pour garantir une parité
mathématique exacte (mêmes fonctions de gradient et de ratio).
"""
from __future__ import annotations
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infer_output_daily_maps_assim as M  # noqa: E402  (helpers partagés)

# Les 23 features dynamiques du modèle (ordre indifférent : lecture par nom à l'inférence).
FEATURES = [
    "swh", "mwp", "shww", "mpww", "sshf", "slhf", "wind_speed_neutral", "sea_slope",
    "msl", "ssr", "str", "tsr", "analysed_sst", "grad_analysed_sst", "sos", "dos",
    "sss", "grad_sss", "adt", "sla", "grad_sla", "ugos", "vgos",
]


def derive_features(d: netCDF4.Dataset, lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    def R(name: str) -> np.ndarray:
        # read_merged ignore valid_min/valid_max (faux dans les merged v2 repackes) et
        # ne masque que sur _FillValue -> (24, H, W) float32, manquant -> NaN
        return M.read_merged(d, name)

    feats: dict[str, np.ndarray] = {}
    for f in FEATURES:
        if f == "wind_speed_neutral":
            feats[f] = np.sqrt(R("u10n") ** 2 + R("v10n") ** 2).astype(np.float32)
        elif f == "sea_slope":
            feats[f] = M.safe_ratio(R("swh"), R("mwp")).astype(np.float32)
        elif f == "grad_analysed_sst":
            feats[f] = M.gradient_magnitude_per_m(R("analysed_sst"), lat, lon)  # K/m
        elif f == "grad_sss":
            feats[f] = M.gradient_magnitude_per_m(R("sos"), lat, lon)  # psu/m
        elif f == "sss":
            feats[f] = R("sos") 
        else:
            feats[f] = R(f)
    return feats


def build_day(in_path: Path, out_path: Path, complevel: int) -> dict:
    with netCDF4.Dataset(in_path) as d:
        lat = M.to_numpy(d.variables["latitude"][:])
        lon = M.to_numpy(d.variables["longitude"][:])
        tim = np.asarray(d.variables["time"][:])
        tim_units = getattr(d.variables["time"], "units", "")
        tim_cal = getattr(d.variables["time"], "calendar", "standard")
        feats = derive_features(d, lat, lon)
    nlat, nlon = lat.size, lon.size
    tmp = out_path.with_suffix(".tmp.nc")
    with netCDF4.Dataset(tmp, "w") as o:
        o.createDimension("time", tim.size)
        o.createDimension("latitude", nlat)
        o.createDimension("longitude", nlon)
        vt = o.createVariable("time", "f8", ("time",))
        vt.units = tim_units
        vt.calendar = tim_cal
        vt[:] = tim
        o.createVariable("latitude", "f4", ("latitude",))[:] = lat
        o.createVariable("longitude", "f4", ("longitude",))[:] = lon
        for name, arr in feats.items():
            v = o.createVariable(
                name, "f4", ("time", "latitude", "longitude"),
                zlib=complevel > 0, complevel=complevel,
                chunksizes=(tim.size, nlat, nlon), fill_value=np.float32(np.nan),
            )
            v[:] = arr.astype(np.float32)
        o.source = f"slim input cache from {in_path.name}; 23 derived dynamic features; sss=sos"
    tmp.replace(out_path)
    finite = float(np.isfinite(feats["sos"]).mean())
    return {"finite_sos": round(finite, 3), "mb": round(out_path.stat().st_size / 1e6, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="output_daily")
    p.add_argument("--out-dir", default="output_daily_cache")
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--complevel", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--map-prefix", default="merged_", help="prefixe des cartes journalieres (glorys_merged_ pour GLORYS)")
    args = p.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = M.discover_daily_files(in_dir, prefix=args.map_prefix)
    print(files)
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    days = [day for day in sorted(files) if start <= day <= end]
    print({"to_build": len(days), "start": str(start), "end": str(end)}, flush=True)
    done = 0
    skipped: list[str] = []
    for day in days:
        out = out_dir / f"cache_{day.strftime('%Y%m%d')}.nc"
        if out.exists() and not args.overwrite:
            done += 1
            continue
        try:
            info = build_day(files[day], out, args.complevel)
        except Exception as exc:
            # fichier map corrompu / incomplet (variable absente, HDF error...) -> on saute et on log,
            # au lieu de tuer toute la construction du cache.
            out.with_suffix(".tmp.nc").unlink(missing_ok=True)
            skipped.append(str(day))
            print({"day": str(day), "status": "SKIP", "file": files[day].name,
                   "error": f"{type(exc).__name__}: {str(exc)[:140]}"}, flush=True)
            continue
        done += 1
        print({"day": str(day), **info, "progress": f"{done}/{len(days)}"}, flush=True)
    if skipped:
        print({"WARNING": f"{len(skipped)} jour(s) SKIPPE(S) (fichier corrompu/incomplet)", "days": skipped}, flush=True)
    print({"built": done, "skipped": len(skipped), "out_dir": str(out_dir)}, flush=True)


if __name__ == "__main__":
    main()
