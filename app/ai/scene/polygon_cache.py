"""
Polygon Cache: Caches computed polygons to avoid recomputation.

Prepared for future multithreading by using thread-safe operations.
"""

import logging
from typing import Optional, Dict
from threading import Lock
from collections import OrderedDict

from app.ai.config import get_detection_config
from app.core.logger import setup_logging

from .roof_plane import RoofPlane, RoofPlanePolygon

logger: logging.Logger = setup_logging()


class PolygonCache:
    """
    Thread-safe cache for computed polygons.
    
    Avoids recomputing contours and polygon simplification for the same masks.
    Uses LRU (Least Recently Used) eviction policy.
    
    Configuration from detection.yaml → scene_performance:
    - enable_polygon_cache: Whether to enable caching
    - cache_max_size: Maximum number of cached polygons
    """
    
    def __init__(self):
        """Initialize the polygon cache."""
        self.config = get_detection_config()
        self.cache_config = self.config.scene_understanding.scene_performance
        self.enabled = self.cache_config.enable_polygon_cache
        self.max_size = self.cache_config.cache_max_size
        
        # LRU cache using OrderedDict
        self.cache: Dict[str, RoofPlanePolygon] = OrderedDict()
        self.lock = Lock()  # For thread safety
        
        logger.info(f"PolygonCache initialized (enabled={self.enabled}, max_size={self.max_size})")
    
    def get(self, plane_id: str) -> Optional[RoofPlanePolygon]:
        """
        Retrieve a cached polygon.
        
        Args:
            plane_id: Unique ID of the plane
        
        Returns:
            Cached RoofPlanePolygon or None if not found
        """
        if not self.enabled:
            return None
        
        with self.lock:
            if plane_id in self.cache:
                # Move to end (mark as recently used)
                self.cache.move_to_end(plane_id)
                return self.cache[plane_id]
            return None
    
    def set(self, plane_id: str, polygon: RoofPlanePolygon) -> bool:
        """
        Cache a polygon.
        
        Args:
            plane_id: Unique ID of the plane
            polygon: RoofPlanePolygon to cache
        
        Returns:
            True if cached successfully, False if caching disabled
        """
        if not self.enabled:
            return False
        
        with self.lock:
            # If already in cache, remove to update position
            if plane_id in self.cache:
                del self.cache[plane_id]
            
            # Add to cache
            self.cache[plane_id] = polygon
            
            # Evict LRU item if over limit
            if len(self.cache) > self.max_size:
                oldest_id = next(iter(self.cache))
                del self.cache[oldest_id]
                logger.debug(f"Evicted polygon {oldest_id} from cache (size={len(self.cache)})")
            
            return True
    
    def cache_plane(self, plane: RoofPlane) -> bool:
        """
        Cache the polygon from a plane.
        
        Args:
            plane: RoofPlane with computed polygon
        
        Returns:
            True if cached successfully
        """
        if plane.polygon is not None:
            return self.set(plane.id, plane.polygon)
        return False
    
    def clear(self):
        """Clear the entire cache."""
        with self.lock:
            self.cache.clear()
            logger.info("PolygonCache cleared")
    
    def get_size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / max(self.max_size, 1),
            }
    
    def disable(self):
        """Disable caching."""
        self.enabled = False
        self.clear()
        logger.info("PolygonCache disabled")
    
    def enable(self):
        """Enable caching."""
        self.enabled = True
        logger.info("PolygonCache enabled")
