"""
RoofMeasurementExtractor
========================

Extracts real-world measurements from YOLO segmentation results and
converts them into 3D-ready geometry using image calibration.

Pipeline:
    YOLODetectionResult
        ↓
    _extract_polygon()      — maska → zjednodušený polygón (pixel coords)
        ↓
    _measure_plane()        — pixel rozmery + bbox + area
        ↓
    _apply_calibration()    — px → metre (CalibrationModel)
        ↓
    _estimate_z()           — Z výška z triedy sklonu + ridge height
        ↓
    RoofPlane (3D)          — polygon2D + slope + height_at_vertices
        ↓
    RoofGeometry            — 3D model pripravený na export

Kalibrácia:
    Použij CalibrationService.calibrate_from_distance() s dvoma bodmi
    z Google Maps meradla (napr. 25.61 m = N pixelov).
    Alebo ak poznáš GSD (Ground Sample Distance) satelitu, použi
    CalibrationModel.from_gsd() factory (viď nižšie).
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np

from app.ai.services.yolo_detector import YOLODetectionResult, SegmentationMask, DetectionBox
from app.geometry.calibration import CalibrationModel, CalibrationService
from app.geometry.point import Point2D, Point3D
from app.geometry.polygon import Polygon2D
from app.geometry.plane import RoofPlane

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slope lookup — mapuje YOLO class_name na typický sklon v stupňoch
# Prispôsob podľa svojho datasetu ak treba.
# ---------------------------------------------------------------------------
_SLOPE_DEGREES: Dict[str, float] = {
    "slope_flat":  5.0,   # takmer rovná strecha
    "slope_min":  15.0,   # mierny sklon
    "slope_poly": 25.0,   # polygonálna (zložená) rovina
    "slope_trap": 35.0,   # lichobežníkový rez — typická valbová strecha
    "slope_trop": 35.0,   # alias pre slope_trap (YOLO model používa "trop")
    "slope_tri":  40.0,   # trojuholníkový rez — štítová stena
    # fallback pre staré triedy
    "roof_plane": 30.0,
    "roof_area":  30.0,
}


# ---------------------------------------------------------------------------
# Výstupný dataclass pre jednu nameranú rovinu
# ---------------------------------------------------------------------------
@dataclass
class PlaneMeasurement:
    """
    Pixelové aj reálne rozmery jednej strešnej roviny detegovanej YOLO.

    Atribúty:
        plane_id        : unikátny ID roviny
        class_name      : YOLO trieda (napr. 'slope_trap')
        confidence      : skóre detekcie

        # --- pixelové rozmery ---
        polygon_px      : zjednodušený polygón vo vstupných pixeloch [(x,y), ...]
        bbox_px         : (x_min, y_min, x_max, y_max) v pixeloch
        area_px         : plocha masky v pixeloch²
        width_px        : šírka bounding boxu v pixeloch
        height_px       : výška bounding boxu v pixeloch
        perimeter_px    : obvod polygónu v pixeloch

        # --- reálne rozmery (metre) ---
        polygon_m       : polygón v metroch (len ak je kalibrácia)
        area_m2         : plocha v m²
        width_m         : šírka bbox v metroch
        height_m        : výška bbox v metroch
        perimeter_m     : obvod v metroch
        true_area_m2    : skutočná plocha povrchu (s korekciou sklonu)

        # --- 3D parametre ---
        slope_deg       : odhadovaný sklon podľa triedy
        ridge_height_m  : výška hrebeňa nad okapom (vstup užívateľa alebo 0)
        height_at_verts : Z-súradnica pre každý vrchol polygónu (v metroch)

        # --- hotový RoofPlane (pre RoofGeometry) ---
        roof_plane      : app.geometry.plane.RoofPlane pripravený na 3D export
    """
    plane_id: str
    class_name: str
    confidence: float

    # pixely
    polygon_px: List[Tuple[float, float]] = field(default_factory=list)
    bbox_px: Tuple[float, float, float, float] = (0, 0, 0, 0)
    area_px: float = 0.0
    width_px: float = 0.0
    height_px: float = 0.0
    perimeter_px: float = 0.0

    # metre (None ak chýba kalibrácia)
    polygon_m: Optional[List[Tuple[float, float]]] = None
    area_m2: Optional[float] = None
    width_m: Optional[float] = None
    height_m: Optional[float] = None
    perimeter_m: Optional[float] = None
    true_area_m2: Optional[float] = None

    # 3D
    slope_deg: float = 30.0
    ridge_height_m: float = 0.0
    height_at_verts: List[float] = field(default_factory=list)

    # finálny RoofPlane
    roof_plane: Optional[RoofPlane] = None

    def summary(self) -> str:
        lines = [
            f"[{self.plane_id}] {self.class_name}  conf={self.confidence:.2f}",
            f"  bbox px:  {self.width_px:.0f} × {self.height_px:.0f}  area={self.area_px:.0f} px²",
        ]
        if self.area_m2 is not None:
            lines.append(
                f"  real:     {self.width_m:.2f} × {self.height_m:.2f} m  "
                f"area={self.area_m2:.2f} m²  true_area={self.true_area_m2:.2f} m²"
            )
        lines.append(f"  slope:    {self.slope_deg:.1f}°  ridge_h={self.ridge_height_m:.2f} m")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hlavný extraktor
# ---------------------------------------------------------------------------
class RoofMeasurementExtractor:
    """
    Konvertuje YOLO výstupy na pixelové + reálne rozmery + 3D RoofPlane objekty.

    Použitie:
        extractor = RoofMeasurementExtractor(
            calibration=calib,          # voliteľné
            ridge_height_m=3.5,         # výška hrebeňa nad okapom
            polygon_epsilon=0.02,       # Douglas-Peucker zjednodušenie (0.0–0.1)
            min_area_px=500,            # filtruj príliš malé detekcie
        )
        measurements = extractor.extract(yolo_result)
    """

    def __init__(
        self,
        calibration: Optional[CalibrationModel] = None,
        ridge_height_m: float = 0.0,
        polygon_epsilon: float = 0.02,
        min_area_px: int = 500,
        slope_override: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            calibration       : CalibrationModel z CalibrationService.calibrate_from_distance()
                                Ak None, reálne rozmery nebudú vypočítané.
            ridge_height_m    : Výška hrebeňa nad okapom v metroch.
                                Kľúčový vstup pre Z-koordinátu 3D modelu.
                                Ak 0.0, 3D model bude plochý (Z=0 všade) — použiteľné
                                ako základ pre manuálnu úpravu výšky neskôr.
            polygon_epsilon   : Epsilon pre Douglas-Peucker uprosenie (relatívne k obvodu).
                                0.01 = menej vrcholov, 0.005 = viac detailov.
            min_area_px       : Minimálna plocha masky v pixeloch (filtruje šum).
            slope_override    : Voliteľný dict {class_name: degrees} pre vlastné hodnoty sklonu.
        """
        self.calibration = calibration
        self.ridge_height_m = ridge_height_m
        self.polygon_epsilon = polygon_epsilon
        self.min_area_px = min_area_px
        self._slope_table = {**_SLOPE_DEGREES, **(slope_override or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, yolo_result: YOLODetectionResult) -> List[PlaneMeasurement]:
        """
        Extrahovanie meraní zo všetkých masiek v YOLO výsledku.

        Returns:
            Zoznam PlaneMeasurement zoradených podľa plochy (najväčšia prvá).
        """
        measurements = []

        for mask_obj in yolo_result.masks:
            m = self._process_mask(mask_obj)
            if m is not None:
                measurements.append(m)

        # Zoradiť podľa plochy — väčšie roviny sú zaujímavejšie
        measurements.sort(key=lambda m: m.area_px, reverse=True)

        logger.info(
            f"RoofMeasurementExtractor: {len(measurements)} rovín z "
            f"{len(yolo_result.masks)} masiek"
        )
        return measurements

    def extract_single_mask(
        self,
        mask: np.ndarray,
        class_name: str,
        confidence: float,
    ) -> Optional[PlaneMeasurement]:
        """
        Spracovanie jednej binárnej masky (napr. manuálne testovanie).

        Args:
            mask       : 2D binárna maska (H×W, uint8, hodnoty 0/1)
            class_name : YOLO trieda
            confidence : skóre

        Returns:
            PlaneMeasurement alebo None ak maska nevyhoví filtru.
        """
        # Vytvoríme dočasný SegmentationMask objekt
        area = int(mask.sum())
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        box = DetectionBox(
            x1=float(xs.min()), y1=float(ys.min()),
            x2=float(xs.max()), y2=float(ys.max()),
            confidence=confidence, class_id=0, class_name=class_name
        )
        sm = SegmentationMask(mask=mask, box=box, area=area)
        return self._process_mask(sm)

    @classmethod
    def calibration_from_google_maps(
        cls,
        p1_px: Tuple[float, float],
        p2_px: Tuple[float, float],
        real_distance_m: float,
    ) -> CalibrationModel:
        """
        Skratka pre kalibráciu priamo z Google Maps meradla.

        Args:
            p1_px          : pixel (x, y) prvého bodu merania
            p2_px          : pixel (x, y) druhého bodu merania
            real_distance_m: reálna vzdialenosť v metroch (hodnota z Google Maps)

        Returns:
            CalibrationModel

        Príklad (z tvojho screenshotu 25.61 m):
            calib = RoofMeasurementExtractor.calibration_from_google_maps(
                p1_px=(748, 350),    # biely krúžok pri 25.61m
                p2_px=(1110, 583),   # druhý krúžok (koniec meradla)
                real_distance_m=25.61
            )
        """
        return CalibrationService.calibrate_from_distance(
            point1_pixel=Point2D(*p1_px),
            point2_pixel=Point2D(*p2_px),
            real_world_distance_meters=real_distance_m,
            unit="m",
        )

    # ------------------------------------------------------------------
    # Interná logika
    # ------------------------------------------------------------------

    def _process_mask(self, mask_obj: "SegmentationMask") -> Optional[PlaneMeasurement]:
        """Spracuje jednu SegmentationMask → PlaneMeasurement."""
        box = mask_obj.box
        mask = mask_obj.mask

        # --- filter minimálnej plochy ---
        area_px = float(mask.sum())
        if area_px < self.min_area_px:
            logger.debug(f"Maska preskočená: area={area_px:.0f} < min={self.min_area_px}")
            return None

        # --- polygón z masky ---
        polygon_px = self._extract_polygon(mask, area_px)
        if len(polygon_px) < 3:
            logger.debug("Polygón má menej ako 3 vrcholy, preskakujem.")
            return None

        # --- bbox a pixel rozmery ---
        bbox_px = (box.x1, box.y1, box.x2, box.y2)
        width_px = box.x2 - box.x1
        height_px = box.y2 - box.y1
        perimeter_px = self._polygon_perimeter(polygon_px)

        # --- sklon z triedy ---
        slope_deg = self._slope_table.get(box.class_name, 30.0)

        m = PlaneMeasurement(
            plane_id=str(uuid.uuid4())[:8],
            class_name=box.class_name,
            confidence=box.confidence,
            polygon_px=polygon_px,
            bbox_px=bbox_px,
            area_px=area_px,
            width_px=width_px,
            height_px=height_px,
            perimeter_px=perimeter_px,
            slope_deg=slope_deg,
            ridge_height_m=self.ridge_height_m,
        )

        # --- kalibrácia pixel → metre ---
        if self.calibration is not None:
            self._apply_calibration(m)

        # --- Z-súradnice (výška) ---
        self._estimate_z(m, polygon_px)

        # --- RoofPlane pre 3D export ---
        m.roof_plane = self._build_roof_plane(m)

        return m

    def _extract_polygon(
        self, mask: np.ndarray, area_px: float
    ) -> List[Tuple[float, float]]:
        """
        Kontúra z binárnej masky → zjednodušený polygón.

        Používa Douglas-Peucker epsilon relatívny k obvodu,
        nie k ploché — to dáva lepšie výsledky pre rôzne veľkosti striech.
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return []

        # Najväčšia kontúra
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        epsilon = self.polygon_epsilon * perimeter
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        return [(float(p[0][0]), float(p[0][1])) for p in simplified]

    def _apply_calibration(self, m: PlaneMeasurement) -> None:
        """Konvertuje pixelové rozmery na metre pomocou CalibrationModel."""
        px_per_m = self.calibration.scale_factor_pixels_per_meter

        m.polygon_m = [
            (x / px_per_m, y / px_per_m) for x, y in m.polygon_px
        ]
        m.area_m2 = m.area_px / (px_per_m ** 2)
        m.width_m = m.width_px / px_per_m
        m.height_m = m.height_px / px_per_m
        m.perimeter_m = m.perimeter_px / px_per_m

        # Skutočná plocha povrchu (korigovaná sklonom)
        # true_area = projected_area / cos(slope)
        slope_rad = math.radians(m.slope_deg)
        if m.slope_deg < 89.9:
            m.true_area_m2 = m.area_m2 / math.cos(slope_rad)
        else:
            m.true_area_m2 = float("inf")

    def _estimate_z(
        self,
        m: PlaneMeasurement,
        polygon_px: List[Tuple[float, float]],
    ) -> None:
        """
        Odhadne Z-súradnicu (výšku) každého vrcholu polygónu.

        Stratégia:
        - Okapová línia (eave) = Z=0  →  najnižšia Y-hodnota v orto pohľade
          odpovedá okapu, najvyššia Y odpovedá hrebeňu.
        - Lineárna interpolácia výšky medzi okapom a hrebeňom podľa Y pozície.

        Pre reálne výsledky potrebuješ ridge_height_m od užívateľa.
        Ak je 0.0, všetky Z=0 → plochý 3D model (základ pre ďalšiu editáciu).

        Poznámka: Google Maps je orto pohľad zhora (nie perspektívny),
        takže Y-os obrázka ≈ hĺbka strechy v pôdoryse.
        """
        if not polygon_px:
            m.height_at_verts = []
            return

        ys = [v[1] for v in polygon_px]
        y_min, y_max = min(ys), max(ys)
        y_range = y_max - y_min

        if y_range < 1.0 or self.ridge_height_m == 0.0:
            # Plochá strecha alebo žiadna výška — Z=0 všade
            m.height_at_verts = [0.0] * len(polygon_px)
            return

        # Pre valbovú strechu: Y_min → okap (Z=0), stred → hrebeň (Z=ridge_height_m)
        # Predpokladáme symetrickú valbovú strechu
        heights = []
        for (x, y) in polygon_px:
            # Normalizovaná pozícia 0.0 (okap) → 1.0 (stred/hrebeň)
            t = (y - y_min) / y_range          # 0..1 pozdĺž Y-osi
            # Pre symetrickú strechu: vrchol výšky je uprostred (t=0.5)
            t_symmetric = 1.0 - abs(2.0 * t - 1.0)   # 0→0, 0.5→1, 1→0
            z = t_symmetric * self.ridge_height_m
            heights.append(round(z, 4))

        m.height_at_verts = heights

    def _build_roof_plane(self, m: PlaneMeasurement) -> Optional[RoofPlane]:
        """
        Zostaví app.geometry.plane.RoofPlane z PlaneMeasurement.
        Používa reálne metre ak je kalibrácia, inak pixely.
        """
        verts_2d = m.polygon_m if m.polygon_m else m.polygon_px
        if len(verts_2d) < 3:
            return None

        try:
            polygon_2d = Polygon2D(
                vertices=[Point2D(x, y) for x, y in verts_2d]
            )
            return RoofPlane(
                name=f"{m.class_name}_{m.plane_id}",
                polygon=polygon_2d,
                slope=m.slope_deg,
                orientation=0.0,  # TODO: odhadni z orientácie bbox (N/S/E/W)
                height_at_vertices=m.height_at_verts,
            )
        except Exception as e:
            logger.warning(f"Nepodarilo sa zostaviť RoofPlane pre {m.plane_id}: {e}")
            return None

    @staticmethod
    def _polygon_perimeter(polygon: List[Tuple[float, float]]) -> float:
        """Obvod polygónu v pixeloch."""
        if len(polygon) < 2:
            return 0.0
        total = 0.0
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            total += math.hypot(x2 - x1, y2 - y1)
        return total