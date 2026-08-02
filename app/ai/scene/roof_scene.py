"""
RoofScene: Represents a complete roof scene with all detected groups and future extensibility.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime

from .roof_group import RoofGroup


@dataclass
class RoofScene:
    """
    Represents a complete roof scene containing all detected roof groups and auxiliary objects.
    
    This is the primary output of the scene understanding pipeline and serves as a container
    for all detected features. It's designed to be extensible for future object types
    (chimneys, skylights, solar panels, etc.).
    
    Attributes:
        id: Unique scene identifier
        groups: List of RoofGroup objects detected in the image
        image_size: Original image dimensions (width, height)
        timestamp: When this scene was created
        
        # Future extension points (prepared but not implemented):
        features: Dict of future feature types (chimneys, skylights, etc.)
        
        # Statistics
        total_buildings: Number of detected building groups
        total_planes: Total number of roof planes
        average_group_confidence: Average confidence across all groups
        scene_quality_score: Overall scene quality (0.0-1.0)
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    groups: List[RoofGroup] = field(default_factory=list)
    image_size: tuple = (0, 0)  # (width, height)
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Future extension points for other feature types
    # Prepared but not implemented - add as needed:
    # chimneys: List[Chimney] = field(default_factory=list)
    # skylights: List[Skylight] = field(default_factory=list)
    # solar_panels: List[SolarPanel] = field(default_factory=list)
    # vents: List[Vent] = field(default_factory=list)
    # gutters: List[Gutter] = field(default_factory=list)
    
    # Generic feature container for future extensibility
    features: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Statistics (computed on initialization)
    total_buildings: int = 0
    total_planes: int = 0
    average_group_confidence: float = 0.0
    scene_quality_score: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Compute scene-level statistics."""
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute statistics from groups."""
        self.total_buildings = len(self.groups)
        self.total_planes = sum(g.plane_count for g in self.groups)
        
        if self.groups:
            self.average_group_confidence = sum(g.average_confidence for g in self.groups) / len(self.groups)
            self.scene_quality_score = sum(g.group_quality_score for g in self.groups) / len(self.groups)
        else:
            self.average_group_confidence = 0.0
            self.scene_quality_score = 0.0
    
    def add_group(self, group: RoofGroup):
        """Add a group to the scene and update statistics."""
        self.groups.append(group)
        self._compute_statistics()
    
    def remove_group(self, group_id: str) -> bool:
        """Remove a group by ID. Returns True if removed, False if not found."""
        initial_count = len(self.groups)
        self.groups = [g for g in self.groups if g.id != group_id]
        if len(self.groups) < initial_count:
            self._compute_statistics()
            return True
        return False
    
    def get_group_by_id(self, group_id: str) -> Optional[RoofGroup]:
        """Find a group by ID."""
        for group in self.groups:
            if group.id == group_id:
                return group
        return None
    
    def get_plane_by_id(self, plane_id: str) -> Optional[tuple]:
        """
        Find a plane by ID.
        
        Returns:
            Tuple of (group, plane) if found, None otherwise.
        """
        for group in self.groups:
            for plane in group.planes:
                if plane.id == plane_id:
                    return (group, plane)
        return None
    
    def add_feature(self, feature_type: str, feature_object: Any):
        """
        Add a feature object to the scene (for future extensions).
        
        Examples:
            scene.add_feature('chimneys', chimney_obj)
            scene.add_feature('solar_panels', panel_obj)
        """
        if feature_type not in self.features:
            self.features[feature_type] = []
        self.features[feature_type].append(feature_object)
    
    def get_features(self, feature_type: str) -> List[Any]:
        """Get all features of a specific type."""
        return self.features.get(feature_type, [])
    
    def has_features(self, feature_type: str) -> bool:
        """Check if scene has any features of a specific type."""
        return feature_type in self.features and len(self.features[feature_type]) > 0
    
    @property
    def is_empty(self) -> bool:
        """Check if scene has any content."""
        return len(self.groups) == 0 and len(self.features) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scene to dictionary for serialization."""
        return {
            'id': self.id,
            'image_size': self.image_size,
            'timestamp': self.timestamp.isoformat(),
            'total_buildings': self.total_buildings,
            'total_planes': self.total_planes,
            'average_group_confidence': self.average_group_confidence,
            'scene_quality_score': self.scene_quality_score,
            'groups': [
                {
                    'id': g.id,
                    'plane_count': g.plane_count,
                    'average_confidence': g.average_confidence,
                    'total_area_pixels': g.total_area_pixels,
                    'group_quality_score': g.group_quality_score,
                }
                for g in self.groups
            ],
            'features': {k: len(v) for k, v in self.features.items()},
            'metadata': self.metadata,
        }
