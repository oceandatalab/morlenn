#!/usr/bin/env python
"""build_coords.py 
--> Build positions file (lat, lon, day, mld) in the same order as precompute:
Monthly netCDF Dataset, with non-valid station filtered, incomplete month skipped

USAGE:
  build_coords.py --config CONFIG --split {train,val} --out coords_<split>.npz

ARGUMENTS: 
    --config configuration file
    --split Either train dataset or val dataset
    --out Name of coords file in numpy npz format

RETURNS
    RETURNS the coordinate in the same order as the precompute file
"""
import argparse
import datetime
import numpy as np
import netCDF4

from morlenn.config import load_config
from morlenn.data import MonthlyNetCDFDataset, list_netcdf_files


def ordinal(y, m, d):
    try:
        return float(datetime.date(int(y), int(m), int(d)).toordinal())
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=("train", "val"), default=None,
                    help="Legacy per-split mode (reads config train_dir/val_dir).")
    ap.add_argument("--input-dir", default=None,
                    help="Global mode: directory of NetCDF months (e.g. data/mld_split/all).")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.input_dir is not None:
        label = "all"
        data_dir = a.input_dir
    else:
        if a.split is None:
            raise SystemExit("Provide either --split {train,val} or --input-dir (global mode).")
        label = a.split
        data_dir = cfg.paths.train_dir if a.split == "train" else cfg.paths.val_dir
    ds = MonthlyNetCDFDataset(
        files=list_netcdf_files(data_dir),
        dynamic_features=cfg.dataset.dynamic_features,
        static_features=cfg.dataset.static_features,
        targets=cfg.dataset.targets,
        mld_target_transform=cfg.dataset.mld_target_transform,
        stats=None,  # does not affect index / order of samples
        replace_nan_with_zero=cfg.dataset.replace_nan_with_zero,
        clip_mld_max=cfg.dataset.clip_mld_max,
        dynamic_history_hours=cfg.dataset.dynamic_history_hours,
        strict_static_features=cfg.dataset.strict_static_features,
        eddy_max_distance_radius=cfg.dataset.eddy_max_distance_radius,
    )
    N = len(ds)
    lat = np.empty(N)
    lon = np.empty(N)
    day = np.empty(N)
    mld = np.empty(N)
    year = np.empty(N)
    month = np.empty(N)
    cache = {}
    for i in range(N):
        _, ppath, sidx = ds._locate(i)
        c = cache.get(ppath)
        if c is None:
            d = netCDF4.Dataset(ppath, "r")
            c = {k: np.array(d.variables[k][:]).ravel() for k in ["lat", "lon", "year", "month", "day", "mld"]}
            d.close()
            cache[ppath] = c
        lat[i] = c["lat"][sidx]; lon[i] = c["lon"][sidx]; mld[i] = c["mld"][sidx]
        year[i] = c["year"][sidx]; month[i] = c["month"][sidx]
        day[i] = ordinal(c["year"][sidx], c["month"][sidx], c["day"][sidx])
        if i and i % 50000 == 0:
            print(f"  {label} {i}/{N}", flush=True)
    np.savez(a.out, lat=lat, lon=lon, day=day, mld=mld, year=year, month=month)
    print(f"coords {label}: N={N} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
