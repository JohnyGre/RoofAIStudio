"""
Výpočet plochy a základných metrík strešného polygónu.
Shoelace formula (Gaussov vzorec) pre plochu, dĺžka obvodu, ťažisko.
"""
from dataclasses import dataclass
from typing import List, Tuple

from app.geometry.point import Point2D # Používame existujúcu dataclass Point2D

@dataclass
class RoofPolygonMetrics:
    area_m2: float
    perimeter_m: float
    centroid: Point2D # Centroid ako Point2D dataclass
    vertex_count: int


class RoofAreaCalculator:
    """Bezstavová služba — čistá geometria, žiadna závislosť na UI/DB."""

    @staticmethod
    def calculate(points: List[Point2D]) -> RoofPolygonMetrics:
        if len(points) < 3:
            raise ValueError("Polygón potrebuje aspoň 3 body.")

        area = RoofAreaCalculator._shoelace_area(points)
        perimeter = RoofAreaCalculator._perimeter(points)
        centroid_tuple = RoofAreaCalculator._centroid(points, area)
        centroid_point = Point2D(centroid_tuple[0], centroid_tuple[1]) # Konvertujeme na Point2D dataclass

        return RoofPolygonMetrics(
            area_m2=round(area, 2),
            perimeter_m=round(perimeter, 2),
            centroid=centroid_point,
            vertex_count=len(points),
        )

    @staticmethod
    def _shoelace_area(points: List[Point2D]) -> float:
        n = len(points)
        total = 0.0
        for i in range(n):
            x1, y1 = points[i].x, points[i].y
            x2, y2 = points[(i + 1) % n].x, points[(i + 1) % n].y
            total += x1 * y2 - x2 * y1
        return abs(total) / 2.0

    @staticmethod
    def _perimeter(points: List[Point2D]) -> float:
        n = len(points)
        total = 0.0
        for i in range(n):
            x1, y1 = points[i].x, points[i].y
            x2, y2 = points[(i + 1) % n].x, points[(i + 1) % n].y
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total

    @staticmethod
    def _centroid(points: List[Point2D], area: float) -> Tuple[float, float]:
        n = len(points)
        cx = cy = 0.0
        for i in range(n):
            x1, y1 = points[i].x, points[i].y
            x2, y2 = points[(i + 1) % n].x, points[(i + 1) % n].y
            cross = x1 * y2 - x2 * y1
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        factor = 1 / (6 * area) if area else 0
        return (cx * factor, cy * factor)
