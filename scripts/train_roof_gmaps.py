"""
Fine-tune YOLOv8n-seg for Google Maps roof detection.
RTX 3050 4GB, CUDA 11.8, Python 3.13.
"""
import os, sys, shutil
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available(), "CUDA not available!"
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory//1024**3}GB)")

    model = YOLO("ai_models/roof_finetuned.pt")
    results = model.train(
        data="data/datasets/roof-1/data.yaml",
        epochs=120, imgsz=640, batch=8, device=0, workers=0,
        name="roof_gmaps_v2", exist_ok=True,
        optimizer="AdamW", lr0=0.0005, lrf=0.01, warmup_epochs=3,
        degrees=15, shear=5, perspective=0.0005, fliplr=0.5,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, scale=0.5,
        mosaic=0.8, mixup=0.1, copy_paste=0.1, erasing=0.2,
        dropout=0.1, patience=25, save=True, val=True, plots=True,
    )
    src = "runs/segment/roof_gmaps_v2/weights/best.pt"
    if os.path.exists(src):
        shutil.copy(src, "ai_models/roof_gmaps_v2.pt")
        m = YOLO(src)
        metrics = m.val(data="data/datasets/roof-1/data.yaml", verbose=False)
        print(f"mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")
    print("Done")

if __name__ == "__main__":
    main()
