# -*- coding: utf-8 -*-
"""
Obnovenie (resume) prerušeného tréningu YOLO-seg.

Použitie:
    python tools/resume_training.py roof_mega_s
    python tools/resume_training.py roof_final_v3 --epochs 40

Skript:
  1. Skontroluje, že existuje ai_models/runs/<názov>/weights/last.pt
  2. Vypíše aktuálny stav z results.csv (dokončené epochy, posledné mAP)
  3. Obnoví tréning z last.pt (Ultralytics resume — parametre sa načítajú
     z args.yaml uloženého v run priečinku; počet epoch sa dá navýšiť cez --epochs)
"""

import argparse
import csv
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(PROJ, "ai_models", "runs")


def show_status(run_name: str) -> int:
    """Vráti počet dokončených epoch z results.csv (alebo 0)."""
    csv_path = os.path.join(RUNS, run_name, "results.csv")
    if not os.path.exists(csv_path):
        print(f"  (results.csv zatiaľ neexistuje — tréning ešte nedokončil epochu 1)")
        return 0
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    done = len(rows) - 1
    if done > 0:
        last = rows[-1]
        try:
            box_m = last[9]
            mask_m = last[13]
            print(f"  Dokončené epochy: {done}")
            print(f"  Posledné mAP50:   box={box_m}  mask={mask_m}")
        except IndexError:
            print(f"  Dokončené epochy: {done} (stĺpce mAP nedostupné)")
    return done


def main():
    ap = argparse.ArgumentParser(description="Obnovenie prerušeného tréningu YOLO-seg")
    ap.add_argument("run_name", help="názov runu, napr. roof_mega_s")
    ap.add_argument("--epochs", type=int, default=0,
                    help="celkový počet epoch (default: ponechať z args.yaml)")
    args = ap.parse_args()

    run_dir = os.path.join(RUNS, args.run_name)
    last_pt = os.path.join(run_dir, "weights", "last.pt")

    if not os.path.isdir(run_dir):
        print(f"❌ Run neexistuje: {run_dir}")
        sys.exit(1)
    if not os.path.exists(last_pt):
        print(f"❌ last.pt neexistuje: {last_pt}")
        print("   (resume je možný len ak tréning aspoň raz uložil váhy)")
        sys.exit(1)

    print(f"=== Obnovenie tréningu: {args.run_name} ===")
    print(f"  last.pt: {last_pt}")
    done = show_status(args.run_name)

    if args.epochs:
        print(f"  Celkový cieľ epoch: {args.epochs} (zostáva {args.epochs - done})")
    else:
        print("  Počet epoch ponechaný z args.yaml")

    from ultralytics import YOLO
    model = YOLO(last_pt)
    kwargs = {"resume": True}
    if args.epochs:
        kwargs["epochs"] = args.epochs
    model.train(**kwargs)


if __name__ == "__main__":
    main()
