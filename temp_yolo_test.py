import cv2
from ultralytics import YOLO
from pathlib import Path
import torch

# --- Configuration ---
MODEL_PATH = "ai_models/roof_finetuned.pt"
IMAGE_PATH = "debug_map_image.png"
OUTPUT_PATH = "yolo_only_segmentation_result.png"

def run_yolo_segmentation():
    """
    Runs YOLO segmentation on a single image using a trained model
    and saves the result.
    """
    # Check if files exist
    model_file = Path(MODEL_PATH)
    image_file = Path(IMAGE_PATH)

    if not model_file.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    if not image_file.exists():
        print(f"Error: Image file not found at {IMAGE_PATH}")
        print("Please run the main application first to generate the debug image.")
        return

    print(f"Torch CUDA available: {torch.cuda.is_available()}")
    print("Loading YOLO model...")
    model = YOLO(model_file)

    print(f"Loading image: {image_file}")
    img = cv2.imread(str(image_file))

    print("Running model prediction...")
    # Run prediction
    results = model.predict(img, conf=0.25)

    print("Prediction complete. Plotting results...")
    # Check if there are any results to plot
    if results and results[0].masks is not None:
        # Plot the results on the image
        # .plot() returns a BGR numpy array
        result_image = results[0].plot(boxes=False) # We only want to see the segmentation mask

        # Save the output image
        cv2.imwrite(OUTPUT_PATH, result_image)
        print(f"Successfully saved YOLO-only segmentation to: {OUTPUT_PATH}")
    else:
        print("No objects were detected by the YOLO model in the image.")

if __name__ == "__main__":
    run_yolo_segmentation()