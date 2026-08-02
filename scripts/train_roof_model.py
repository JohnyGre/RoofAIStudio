"""
YOLO Roof Fine-Tuning Pipeline
===============================
Downloads a Roboflow dataset, fine-tunes YOLOv8-seg, and exports the model.

Usage:
    python scripts/train_roof_model.py --dataset-url "clearspot-ank3o/roof-utb0w/1"
    
Or with a local dataset:
    python scripts/train_roof_model.py --data path/to/data.yaml

Requirements:
    pip install roboflow ultralytics
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    format: str = "yolov8",
    output_dir: Optional[str] = None,
) -> str:
    """
    Download a dataset from Roboflow Universe.
    
    Args:
        workspace: Roboflow workspace name (e.g. "clearspot-ank3o")
        project: Project name (e.g. "roof-utb0w")
        version: Dataset version number
        api_key: Roboflow API key
        format: Export format ("yolov8", "yolov8-seg", "coco", etc.)
        output_dir: Where to save (default: data/datasets/)
    
    Returns:
        Path to the downloaded dataset directory
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "roboflow", "-q"])
        from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    
    workspace_obj = rf.workspace(workspace)
    project_obj = workspace_obj.project(project)
    version_obj = project_obj.version(version)
    
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "datasets")
    
    dataset = version_obj.download(format, location=output_dir)
    print(f"Dataset downloaded to: {dataset.location}")
    return dataset.location


def find_data_yaml(dataset_dir: str) -> Optional[str]:
    """Find data.yaml in the downloaded dataset."""
    for root, dirs, files in os.walk(dataset_dir):
        for f in files:
            if f == "data.yaml":
                return os.path.join(root, f)
    return None


def train_model(
    data_yaml: str,
    base_model: str = "yolov8s-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    project_name: str = "roof_training",
    **kwargs,
) -> str:
    """
    Fine-tune YOLOv8 segmentation model on a roof dataset.
    
    Args:
        data_yaml: Path to data.yaml
        base_model: Base model to fine-tune (yolov8n-seg, yolov8s-seg, yolov11s-seg, etc.)
        epochs: Number of training epochs
        imgsz: Image size
        batch: Batch size
        device: "cpu", "cuda", "0", etc.
        project_name: Name for the training run
    
    Returns:
        Path to the best trained model weights
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
        from ultralytics import YOLO

    # Download base model if not exists
    if not os.path.exists(base_model):
        print(f"Base model {base_model} not found locally. YOLO will auto-download it.")

    model = YOLO(base_model)

    print(f"\n{'='*60}")
    print(f"TRAINING: {base_model} on {data_yaml}")
    print(f"Epochs: {epochs} | Image size: {imgsz} | Batch: {batch} | Device: {device}")
    print(f"{'='*60}\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project_name,
        name=f"roof_{Path(data_yaml).parent.name}",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        # Data augmentation (important for roof detection)
        hsv_h=0.015,    # Slight hue shift
        hsv_s=0.4,      # Moderate saturation shift 
        hsv_v=0.2,      # Slight brightness shift
        degrees=10.0,   # Rotate up to 10 degrees
        translate=0.1,  # Translate up to 10%
        scale=0.3,      # Scale up to 30%
        shear=2.0,      # Shear up to 2 degrees
        perspective=0.0,
        flipud=0.5,     # Vertical flip (roofs can be upside down in aerial)
        fliplr=0.5,     # Horizontal flip
        mosaic=0.5,     # Mosaic augmentation
        mixup=0.1,      # MixUp augmentation
        **kwargs,
    )

    # Find best model
    best_pt = str(Path(results.save_dir) / "weights" / "best.pt")
    if os.path.exists(best_pt):
        print(f"\nBest model saved to: {best_pt}")
    else:
        print(f"\nTraining completed. Check: {results.save_dir}")
    
    return best_pt


def export_model(model_path: str, formats: list = None) -> dict:
    """
    Export trained model to various formats.
    
    Args:
        model_path: Path to best.pt
        formats: List of formats ("onnx", "tflite", "torchscript", "openvino", "tensorrt")
    
    Returns:
        Dict of format -> output path
    """
    if formats is None:
        formats = ["onnx"]

    try:
        from ultralytics import YOLO
    except ImportError:
        from ultralytics import YOLO

    model = YOLO(model_path)
    exported = {}

    for fmt in formats:
        try:
            result = model.export(format=fmt, dynamic=False if fmt == "tflite" else True)
            exported[fmt] = result
            print(f"Exported to {fmt}: {result}")
        except Exception as e:
            print(f"Export to {fmt} failed: {e}")

    return exported


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Roof Detection Fine-Tuning Pipeline"
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        default="clearspot-ank3o/roof-utb0w/1",
        help="Roboflow dataset in format 'workspace/project/version'",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("ROBOFLOW_API_KEY", ""),
        help="Roboflow API key (or set ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to local data.yaml (skips Roboflow download)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolov8s-seg.pt",
        help="Base YOLO model (yolov8n-seg, yolov8s-seg, yolo11s-seg, etc.)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--export", nargs="+", default=None,
                        help="Export formats: onnx tflite openvino")
    parser.add_argument("--output-dir", type=str, 
                        default=str(PROJECT_ROOT / "data" / "datasets"))
    
    args = parser.parse_args()

    # Step 1: Get dataset
    if args.data:
        data_yaml = args.data
        print(f"Using local dataset: {data_yaml}")
    elif args.api_key:
        parts = args.dataset_url.strip("/").split("/")
        if len(parts) != 3:
            print(f"Invalid dataset-url format. Expected: workspace/project/version")
            print(f"Got: {args.dataset_url}")
            sys.exit(1)
        
        workspace, project, version = parts
        version = int(version)
        
        print(f"Downloading dataset: {workspace}/{project} v{version}")
        dataset_dir = download_roboflow_dataset(
            workspace, project, version, args.api_key,
            output_dir=args.output_dir,
        )
        data_yaml = find_data_yaml(dataset_dir)
        if not data_yaml:
            print(f"ERROR: No data.yaml found in {dataset_dir}")
            print("Contents:", os.listdir(dataset_dir))
            sys.exit(1)
    else:
        print("ERROR: You need either --data (local dataset) or --api-key (Roboflow).")
        print()
        print("To get a Roboflow API key:")
        print("1. Sign up at https://roboflow.com")
        print("2. Go to Settings -> API Keys")
        print("3. Copy your private API key")
        print("4. Run: python scripts/train_roof_model.py --api-key YOUR_KEY")
        print()
        print("Or pass it via environment: set ROBOFLOW_API_KEY=YOUR_KEY")
        sys.exit(1)

    print(f"Data YAML: {data_yaml}")

    # Step 2: Train
    model_path = train_model(
        data_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # Step 3: Copy best model to ai_models/
    best_dest = str(PROJECT_ROOT / "ai_models" / "roof_finetuned.pt")
    os.makedirs(os.path.dirname(best_dest), exist_ok=True)
    if os.path.exists(model_path):
        shutil.copy2(model_path, best_dest)
        print(f"\nModel copied to: {best_dest}")

    # Step 4: Export (optional)
    if args.export:
        export_model(model_path, formats=args.export)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model: {best_dest}")
    print("=" * 60)
    print("\nTo use the model:")
    print(f"  detector = HybridRoofDetector()")
    print(f"  detector.load_yolo('{best_dest}')")
    print(f"  results = detector.detect(image)")


if __name__ == "__main__":
    main()
