"""
This module defines the RoofGenome data model for summarizing roof geometry characteristics.
"""

from dataclasses import dataclass, field
from typing import Optional
import uuid
import numpy as np # Import numpy for feature_vector type hint

@dataclass(frozen=True)
class RoofGenome:
    """
    A data model representing a 'genome' or signature of a roof's geometry.
    It summarizes key structural characteristics for comparison, classification,
    and AI model input.
    Immutable for safe use.
    """
    # Unique identifier for the genome instance
    id: uuid.UUID = field(default_factory=uuid.uuid4)

    number_of_planes: int = 0
    number_of_edges: int = 0
    number_of_ridges: int = 0
    number_of_valleys: int = 0
    number_of_hips: int = 0 # Added hip_count for consistency with ORM
    number_of_openings: int = 0
    average_slope: float = 0.0
    symmetry_score: float = 0.0 # Added symmetry_score for consistency with ORM
    complexity_score: float = 0.0
    
    # Feature vector for AI similarity search (e.g., embedding from a neural network)
    feature_vector: Optional[np.ndarray] = None # Stored as numpy array in domain model

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
        if self.number_of_hips < 0:
            raise ValueError("Number of hips cannot be negative.")
        if self.number_of_openings < 0:
            raise ValueError("Number of openings cannot be negative.")
        if not (0 <= self.average_slope <= 90):
            raise ValueError("Average slope must be between 0 and 90 degrees.")
        if not (0 <= self.symmetry_score <= 1): # Assuming symmetry score is between 0 and 1
            raise ValueError("Symmetry score must be between 0 and 1.")
        if self.complexity_score < 0:
            raise ValueError("Complexity score cannot be negative.")
        if self.feature_vector is not None and not isinstance(self.feature_vector, np.ndarray):
            raise TypeError("Feature vector must be a NumPy array.")

    def to_dict(self) -> dict:
        """
        Converts the RoofGenome to a dictionary.
        """
        data = self.__dict__.copy()
        if isinstance(data['id'], uuid.UUID):
            data['id'] = str(data['id'])
        if isinstance(data['feature_vector'], np.ndarray):
            data['feature_vector'] = data['feature_vector'].tolist() # Convert numpy array to list for JSON serialization
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RoofGenome":
        """
        Creates a RoofGenome instance from a dictionary.
        """
        if 'id' in data and isinstance(data['id'], str):
            data['id'] = uuid.UUID(data['id'])
        if 'feature_vector' in data and isinstance(data['feature_vector'], list):
            data['feature_vector'] = np.array(data['feature_vector'], dtype=np.float32)
        return cls(**data)
