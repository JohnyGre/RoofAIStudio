"""
This module defines the RoofGenome data model for summarizing roof geometry characteristics.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass(frozen=True)
class RoofGenome:
    """
    A data model representing a 'genome' or signature of a roof's geometry.
    It summarizes key structural characteristics for comparison, classification,
    and AI model input.
    Immutable for safe use.
    """
    number_of_planes: int = 0
    number_of_edges: int = 0
    number_of_ridges: int = 0
    number_of_valleys: int = 0
    number_of_openings: int = 0
    average_slope: float = 0.0
    complexity_score: float = 0.0
    # Add more features as needed, e.g.,
    # max_height: float = 0.0
    # footprint_area: float = 0.0
    # dominant_orientation: Optional[float] = None

    def __post_init__(self):
        """
        Validates the RoofGenome upon initialization.
        """
        if self.number_of_planes < 0:
            raise ValueError("Number of planes cannot be negative.")
        if self.number_of_edges < 0:
            raise ValueError("Number of edges cannot be negative.")
        if self.number_of_ridges < 0:
            raise ValueError("Number of ridges cannot be negative.")
        if self.number_of_valleys < 0:
            raise ValueError("Number of valleys cannot be negative.")
        if self.number_of_openings < 0:
            raise ValueError("Number of openings cannot be negative.")
        if not (0 <= self.average_slope <= 90):
            raise ValueError("Average slope must be between 0 and 90 degrees.")
        if self.complexity_score < 0:
            raise ValueError("Complexity score cannot be negative.")

    def to_dict(self) -> dict:
        """
        Converts the RoofGenome to a dictionary.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict) -> "RoofGenome":
        """
        Creates a RoofGenome instance from a dictionary.
        """
        return cls(**data)
