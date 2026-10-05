"""morlenn — point d'entrée CLI (console_script).

Dispatcher fin : `morlenn <stage> <profile> [args...]`, délègue à run.sh à la racine du dossier
upperdyn_new. Garde une source de vérité unique (run.sh) tout en offrant une commande installable.
Pour une vraie transition en librairie, chaque étape pourra devenir un sous-module appelable.

  morlenn train assim 3,4
  morlenn preprocess assim
  morlenn --root /chemin/upperdyn_new validate assim
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="morlenn",
        description="Dispatcher upperdyn_new : preprocess/train/infer/validate x baseline/assim.",
    )
    ap.add_argument("stage", choices=["preprocess", "train", "infer", "validate"])
    ap.add_argument("profile", choices=["baseline", "assim"])
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="args transmis tels quels à run.sh (GPUs, checkpoint, maps, dates...)")
    ap.add_argument("--root", default=None, help="racine de upperdyn_new (défaut: auto-détecté)")
    a = ap.parse_args(argv)

    root = Path(a.root) if a.root else Path(__file__).resolve().parents[2]
    run = root / "run.sh"
    if not run.exists():
        sys.exit(f"run.sh introuvable à {run}. Précise --root <dossier upperdyn_new>.")
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    raise SystemExit(subprocess.call(["bash", str(run), a.stage, a.profile, *extra]))


if __name__ == "__main__":
    main()
