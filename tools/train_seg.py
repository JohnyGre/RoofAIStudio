# -*- coding: utf-8 -*-
"""
train_seg.py — Tréning YOLO-seg na auto-labelovanom datasete.

POUŽITIE:
    python tools/train_seg.py --data data/datasets/roofs_v1/data.yaml --epochs 50

Poznámka: na CPU je tréning pomalý — pre test použite --epochs 5.
"""
from __future__ import annotations

import argparse
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ)

import torch  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="YOLO-seg tréning")
    parser.add_argument("--data", default="data/datasets/roofs_v1/data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--model", default="yolov8n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--name", default="roof_seg_v1")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    print(f"torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️ CPU tréning — bude pomalý. Pre rýchly test použite --epochs 5.")

    from ultralytics import YOLO

    # Východiskový model: predtrénovaný yolov8n-seg (COCO) alebo lokálny
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJ, "ai_models", model_path)
    if not os.path.exists(model_path):
        print(f"Model {model_path} neexistuje — použijem predtrénovaný yolov8n-seg.pt")
        model_path = "yolov8n-seg.pt"

    model = YOLO(model_path)
    print(f"Model: {model_path}")

    data_yaml = os.path.join(PROJ, args.data)
    if not os.path.exists(data_yaml):
        print(f"ERROR: {data_yaml} neexistuje!")
        sys.exit(1)

    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        patience=args.patience,
        project=os.path.join(PROJ, "ai_models", "runs"),
        exist_ok=True,
        workers=0,
    )

    # Najlepší model
    best = os.path.join(PROJ, "ai_models", "runs", args.name, "weights", "best.pt")
    print(f"\nNajlepší model: {best}")
    if os.path.exists(best):
        print(f"Veľkosť: {os.path.getsize(best)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
