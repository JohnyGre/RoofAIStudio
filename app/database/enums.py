"""
This module defines various Enums used across the database models for Roof AI Studio.
"""

from enum import Enum

class RoofType(str, Enum):
    """Defines the types of roofs."""
    GABLE = "Gable"
    HIP = "Hip"
    FLAT = "Flat"
    SHED = "Shed"
    GAMBREL = "Gambrel"
    MANSARD = "Mansard"
    DORMER = "Dormer"
    OTHER = "Other"

class RoofStatus(str, Enum):
    """Defines the status of a roof analysis or project phase."""
    DRAFT = "Draft"
    PENDING_REVIEW = "Pending Review"
    APPROVED = "Approved"
    REJECTED = "Rejected"
    COMPLETED = "Completed"

class EstimateStatus(str, Enum):
    """Defines the status of an estimate."""
    DRAFT = "Draft"
    SENT = "Sent"
    ACCEPTED = "Accepted"
    REJECTED = "Rejected"
    INVOICED = "Invoiced"
    PAID = "Paid"

class MaterialUnit(str, Enum):
    """Defines the units for material quantities."""
    SQUARE_FOOT = "sq_ft"
    LINEAR_FOOT = "lin_ft"
    EACH = "each"
    BUNDLE = "bundle"
    GALLON = "gallon"
    POUND = "pound"

class PhotoType(str, Enum):
    """Defines the type or perspective of a roof photo."""
    OVERHEAD = "Overhead"
    ANGLE = "Angle"
    DETAIL = "Detail"
    FRONT = "Front"
    BACK = "Back"
    SIDE = "Side"

class AIModelType(str, Enum):
    """Defines the type of AI model."""
    OBJECT_DETECTION = "Object Detection"
    SEGMENTATION = "Segmentation"
    CLASSIFICATION = "Classification"
    REGRESSION = "Regression"
    GENERATIVE = "Generative"

class EdgeType(str, Enum):
    """Defines the type of an edge in roof geometry."""
    RIDGE = "Ridge"
    HIP = "Hip"
    VALLEY = "Valley"
    EAVE = "Eave"
    RAKE = "Rake"
    GUTTER = "Gutter"
    PERIMETER = "Perimeter"
    UNKNOWN = "Unknown"
