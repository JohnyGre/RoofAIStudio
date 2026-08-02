"""
This module defines data structures for image information within Roof AI Studio.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

@dataclass
class ImageInfo:
    """
    Represents detailed information about an image file.
    """
    file_path: Path
    width: int
    height: int
    format: str
    created_date: Optional[datetime] = None
    camera_info: Optional[Dict[str, Any]] = None
    # Add other relevant metadata fields as needed
