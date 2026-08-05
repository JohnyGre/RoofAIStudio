import cv2
from ultralytics import YOLO
from pathlib import Path
import torch
import os

# --- Configuration --
# Sem vložte cesty k obrázkom, ktoré chcete otestovať
IMAGE_DIR = Path(r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\batch_screenshots")
IMAGE_PATHS = list(IMAGE_DIR.glob("*.png"))

MODEL_PATH = "ai_models/roof_gmaps_v2.pt"
OUTPUT_DIR = "yolo_batch_test_results"


def run_yolo_segmentation_batch():
    """
    Runs YOLO segmentation on a batch of images using a trained model
    and saves the results to a dedicated directory.
    """
    # Create output directory if it doesn't exist
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # --- Model Loading ---
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print("Loading YOLO model...")
    model = YOLO(model_file)
    print("-" * 30)

    # --- Image Processing Loop ---
    for i, image_path_str in enumerate(IMAGE_PATHS):
        image_file = Path(image_path_str)
        output_filename = f"yolo_vystup_{i+1}_{image_file.stem}.png"
        output_path = Path(OUTPUT_DIR) / output_filename

        print(f"Processing image {i+1}/{len(IMAGE_PATHS)}: {image_file.name}")

        if not image_file.exists():
            print(f"  -> Error: Image file not found at {image_path_str}")
            print("-" * 30)
            continue

        img = cv2.imread(str(image_file))

        print("  -> Running model prediction...")
        results = model.predict(img, conf=0.25)

        if results and results[0].masks is not None:
            result_image = results[0].plot(boxes=False)
            cv2.imwrite(str(output_path), result_image)
            print(f"  -> Successfully saved result to: {output_path}")
        else:
            print("  -> No objects were detected in this image.")
        
        print("-" * 30)

    print("Batch processing complete.")

if __name__ == "__main__":
    run_yolo_segmentation_batch()