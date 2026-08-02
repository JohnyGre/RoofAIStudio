"""
Immutable data models for representing the topological relationships
and connectivity graph of a roof structure.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

@dataclass(frozen=True)
class EdgeType(Enum):
    """Deterministic classification of roof edges."""
    UNKNOWN = "unknown"
    RIDGE = "ridge"
    VALLEY = "valley"
    HIP = "hip"
    EAVE = "eave"
    BOUNDARY = "boundary"

@dataclass(frozen=True)
class PlaneAdjacency:
    """Describes a connection between two adjacent planes via shared edges."""
    plane_a_id: str
    plane_b_id: str
    shared_edge_ids: Set[str]

@dataclass(frozen=True)
class EdgeNode:
    """Represents a topological edge with connectivity and classification."""
    edge_id: str
    start: tuple
    end: tuple
    edge_type: EdgeType
    left_plane_id: Optional[str]
    right_plane_id: Optional[str]
    length: float

@dataclass(frozen=True)
class PlaneNode:
    """Represents a roof plane in the topological graph."""
    plane_id: str
    edge_ids: Set[str]
    neighbor_plane_ids: Set[str]

@dataclass(frozen=True)
class ValidationIssue:
    """Represents a single validation error or warning."""
    severity: str # e.g., 'error', 'warning'
    code: str # e.g., 'duplicate_edge', 'orphan_plane'
    message: str
    offending_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RoofTopology:
    """Complete topological model of a roof geometry."""
    planes: List[PlaneNode]
    edges: List[EdgeNode]
    adjacency: List[PlaneAdjacency]
    outer_boundary: List[str]
    validation: List[ValidationIssue]
