from app.services.satellite_fetcher import SatelliteImageFetcher
import cv2
import os

def fetch_and_save_image():
    address = "Levanduľová 4, 917 01 Trnava"
    output_dir = r"C:\Users\jangr\.gemini\antigravity\playground\RoofAIStudio\batch_screenshots"
    output_path = os.path.join(output_dir, "levandulova_4.png")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fetcher = SatelliteImageFetcher(
        cache_dir="data/satellite_cache",
        image_size=640,
        zoom=21,
        backend="playwright",
    )

    print(f"Fetching image for: {address}")
    img, meta = fetcher.fetch_by_address(address, country="Slovakia")

    if img is not None:
        print(f"Image fetched successfully. Saving to: {output_path}")
        cv2.imwrite(output_path, img)
        print("Image saved.")
    else:
        print(f"Failed to fetch image. Error: {meta.get('error', 'Unknown error')}")

if __name__ == "__main__":
    fetch_and_save_image()