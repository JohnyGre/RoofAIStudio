"""
This module provides functionality to extract metadata from image files.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from PIL.ExifTags import TAGS

class ImageMetadataExtractor:
    """
    Extracts metadata from image files using Pillow.
    """

    @staticmethod
    def extract_exif(image_path: Path) -> Dict[str, Any]:
        """
        Extracts EXIF data from an image file.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Dict[str, Any]: A dictionary containing EXIF tags and their values.
                            Returns an empty dictionary if no EXIF data is found or on error.
        """
        exif_data = {}
        try:
            with Image.open(image_path) as img:
                if hasattr(img, '_getexif'):
                    raw_exif = img._getexif()
                    if raw_exif:
                        for tag, value in raw_exif.items():
                            decoded = TAGS.get(tag, tag)
                            exif_data[decoded] = value
        except Exception as e:
            print(f"Warning: Could not extract EXIF from {image_path}: {e}")
        return exif_data

    @staticmethod
    def get_creation_date(image_path: Path) -> Optional[datetime]:
        """
        Attempts to get the creation date of an image from EXIF or file system.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Optional[datetime]: The creation date as a datetime object, or None if not found.
        """
        exif_data = ImageMetadataExtractor.extract_exif(image_path)
        if "DateTimeOriginal" in exif_data:
            try:
                return datetime.strptime(exif_data["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        elif "DateTime" in exif_data:
            try:
                return datetime.strptime(exif_data["DateTime"], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        # Fallback to file system modification time
        try:
            return datetime.fromtimestamp(image_path.stat().st_mtime)
        except Exception:
            return None

    @staticmethod
    def get_camera_info(image_path: Path) -> Optional[Dict[str, Any]]:
        """
        Extracts relevant camera information from EXIF data.

        Args:
            image_path (Path): The path to the image file.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with camera info, or None if not found.
        """
        exif_data = ImageMetadataExtractor.extract_exif(image_path)
        camera_info = {}
        if "Make" in exif_data:
            camera_info["Make"] = exif_data["Make"]
        if "Model" in exif_data:
            camera_info["Model"] = exif_data["Model"]
        if "FocalLength" in exif_data:
            camera_info["FocalLength"] = exif_data["FocalLength"]
        if "FNumber" in exif_data:
            camera_info["FNumber"] = exif_data["FNumber"]
        if "ExposureTime" in exif_data:
            camera_info["ExposureTime"] = exif_data["ExposureTime"]
        if "ISOSpeedRatings" in exif_data:
            camera_info["ISOSpeedRatings"] = exif_data["ISOSpeedRatings"]

        return camera_info if camera_info else None
