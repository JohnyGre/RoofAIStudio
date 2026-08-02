"""
Single Roof Analyzer v2 -- SAM auto + YOLO hybrid + scale bar calibration.

Given an address:
1. Geocode -> fetch satellite image (playwright Google Maps)
2. YOLO finds candidate boxes -> SAM refines masks
3. SAM auto (grid-based) finds additional masks missed by YOLO
4. Merge + deduplicate both sources
5. Extract scale bar -> pixels-per-meter calibration
6. Return best roof planes with real-world dimensions
"""

import logging
import time
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

from app.services.satellite_fetcher import SatelliteImageFetcher

logger = logging.getLogger(__name__)


class SingleRoofAnalyzer:
    """
    Focused roof analysis for a single building.

    v2 improvements:
    - SAM automatic mask generation (finds planes YOLO misses)
    - Scale bar extraction (px/m calibration)
    - Hybrid merge of YOLO+SAM + SAM auto results
    """

    def __init__(
        self,
        cache_dir: str = "data/satellite_cache",
        google_api_key: Optional[str] = None,
        zoom: int = 21,
        yolo_conf: float = 0.25,
        min_mask_area: int = 100,
        device: str = "cpu",
    ):
        self.cache_dir = cache_dir
        self.zoom = zoom
        self.yolo_conf = yolo_conf
        self.min_mask_area = min_mask_area
        self.device = device

        self.fetcher = SatelliteImageFetcher(
            cache_dir=cache_dir,
            google_api_key=google_api_key,
            image_size=640,
            zoom=zoom,
            backend="playwright",
        )
        self._segmenter = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load SAM + YOLO models."""
        try:
            from app.ai.sam_roof_segmenter import SAMRoofSegmenter

            self._segmenter = SAMRoofSegmenter(
                model_type="vit_b",
                yolo_model_path="ai_models/roof_finetuned.pt",
                device=self.device,
            )
            self._segmenter.load()
            self._loaded = True
            logger.info("SingleRoofAnalyzer v2 loaded (SAM + YOLO)")
            return True
        except Exception as e:
            logger.error("Failed to load SingleRoofAnalyzer: %s", e)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Scale Bar Calibration
    # ------------------------------------------------------------------

    def _extract_scale(self, image: np.ndarray) -> Optional[float]:
        """Try to extract px/m from scale bar. Returns None if unavailable."""
        try:
            from app.services.scale_extractor import extract_scale_bar
            result = extract_scale_bar(image, zoom=self.zoom, debug=False)
            if result is not None:
                return result[0]  # px_per_m
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def analyze_address(
        self,
        address: str,
        country: Optional[str] = None,
        force_refresh: bool = False,
        top_n: int = 8,
    ) -> Dict[str, Any]:
        """Full pipeline with SAM auto + scale bar."""
        t0 = time.time()

        if not self._loaded:
            return {"success": False, "error": "Models not loaded. Call load() first."}

        # Step 1: Fetch satellite image
        img, meta = self.fetcher.fetch_by_address(
            address, country=country, force_refresh=force_refresh
        )
        if img is None:
            return {
                "success": False,
                "error": meta.get("error", "Image fetch failed"),
                "address": address,
                "elapsed_s": time.time() - t0,
            }

        # Step 2: Scale bar extraction
        px_per_m = self._extract_scale(img)

        # Step 3: YOLO -> SAM segmentation
        yolo_results = self._segmenter.segment_auto(
            img,
            conf=self.yolo_conf,
            iou=0.45,
            imgsz=640,
            min_mask_area=self.min_mask_area,
        )
        logger.info("YOLO->SAM: %d planes", len(yolo_results))

        # Step 4: SAM auto (grid-based) for planes YOLO missed
        sam_auto_results = self._run_sam_auto(img)
        logger.info("SAM auto: %d additional candidates", len(sam_auto_results))

        # Step 5: Merge both sources, deduplicate
        merged = self._merge_and_dedup(yolo_results, sam_auto_results)

        # Step 6: Pick top planes
        top_planes = merged[:top_n]

        # Step 7: Clean output
        planes_out = []
        for r in top_planes:
            contour = r.get("contour", [])
            area_m2 = None
            if px_per_m is not None and r.get("area_px", 0) > 0:
                area_m2 = round(r["area_px"] / (px_per_m ** 2), 1)
            planes_out.append({
                "class_name": r.get("class_name", "unknown"),
                "yolo_score": float(r.get("score", 0)),
                "sam_score": float(r.get("sam_score", 0)),
                "composite_score": float(r.get("composite_score", 0)),
                "source": r.get("source", "yolo"),
                "area_px": int(r.get("area_px", r.get("area_640", 0))),
                "area_m2": area_m2,
                "contour": contour,
                "vertices": (
                    [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                    if len(contour) > 0 else []
                ),
            })

        best_plane = planes_out[0] if planes_out else None

        # Step 8: Visualization
        vis = self._draw_merged(img, top_planes)

        elapsed = time.time() - t0
        if best_plane:
            logger.info(
                "SingleRoofAnalyzer v2: %s -> %d total, top=%s (%.3f, %dpx%s) in %.1fs",
                address, len(merged),
                best_plane["class_name"], best_plane["sam_score"],
                best_plane["area_px"],
                f", {best_plane['area_m2']}m2" if best_plane.get("area_m2") else "",
                elapsed,
            )

        return {
            "success": True,
            "address": address,
            "lat": meta.get("lat"),
            "lon": meta.get("lon"),
            "image": img,
            "planes": planes_out,
            "best_plane": best_plane,
            "total_planes": len(merged),
            "total_yolo": len(yolo_results),
            "total_sam_auto": len(sam_auto_results),
            "px_per_m": px_per_m,
            "visualization": vis,
            "elapsed_s": elapsed,
            "source": meta.get("backend", "?"),
        }

    def analyze_coords(
        self,
        lat: float,
        lon: float,
        force_refresh: bool = False,
        top_n: int = 8,
    ) -> Dict[str, Any]:
        """Same as analyze_address but from coordinates directly."""
        t0 = time.time()

        if not self._loaded:
            return {"success": False, "error": "Models not loaded. Call load() first."}

        img, meta = self.fetcher.fetch_by_coords(lat, lon, force_refresh=force_refresh)
        if img is None:
            return {"success": False, "error": "Image fetch failed", "elapsed_s": time.time() - t0}

        px_per_m = self._extract_scale(img)

        yolo_results = self._segmenter.segment_auto(
            img, conf=self.yolo_conf, iou=0.45, imgsz=640, min_mask_area=self.min_mask_area,
        )
        sam_auto_results = self._run_sam_auto(img)
        merged = self._merge_and_dedup(yolo_results, sam_auto_results)
        top_planes = merged[:top_n]

        planes_out = []
        for r in top_planes:
            contour = r.get("contour", [])
            area_m2 = None
            if px_per_m is not None and r.get("area_px", 0) > 0:
                area_m2 = round(r["area_px"] / (px_per_m ** 2), 1)
            planes_out.append({
                "class_name": r.get("class_name", "unknown"),
                "yolo_score": float(r.get("score", 0)),
                "sam_score": float(r.get("sam_score", 0)),
                "composite_score": float(r.get("composite_score", 0)),
                "source": r.get("source", "yolo"),
                "area_px": int(r.get("area_px", r.get("area_640", 0))),
                "area_m2": area_m2,
                "contour": contour,
                "vertices": (
                    [(int(pt[0][0]), int(pt[0][1])) for pt in contour]
                    if len(contour) > 0 else []
                ),
            })

        best_plane = planes_out[0] if planes_out else None
        vis = self._draw_merged(img, top_planes)
        elapsed = time.time() - t0

        return {
            "success": True,
            "lat": lat, "lon": lon,
            "image": img,
            "planes": planes_out,
            "best_plane": best_plane,
            "total_planes": len(merged),
            "total_yolo": len(yolo_results),
            "total_sam_auto": len(sam_auto_results),
            "px_per_m": px_per_m,
            "visualization": vis,
            "elapsed_s": elapsed,
            "source": meta.get("backend", "?"),
        }

    # ------------------------------------------------------------------
    # SAM Auto
    # ------------------------------------------------------------------

    def _run_sam_auto(self, image: np.ndarray, points_per_side: int = 28) -> List[Dict]:
        """Run SAM automatic mask generation, return roof-like masks."""
        try:
            from app.services.sam_auto_masks import generate_roof_masks
            masks = generate_roof_masks(
                image,
                points_per_side=points_per_side,
                min_mask_area=self.min_mask_area,
            )
            # Convert to compatible format
            results = []
            for m in masks:
                # Approximate contour
                mask_uint8 = m["mask"].astype(np.uint8)
                cnts, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(cnts, key=cv2.contourArea) if cnts else np.array([])
                results.append({
                    "class_name": "roof_plane",
                    "score": 0.0,       # no YOLO score
                    "sam_score": float(m["score"]),
                    "composite_score": float(m["score"]),
                    "area_px": m["area_px"],
                    "contour": contour,
                    "mask": m["mask"],
                    "source": "sam_auto",
                })
            return results
        except Exception as e:
            logger.warning("SAM auto failed (falling back to YOLO only): %s", e)
            return []

    # ------------------------------------------------------------------
    # Merge + Deduplicate
    # ------------------------------------------------------------------

    def _merge_and_dedup(
        self,
        yolo_results: List[Dict],
        sam_auto_results: List[Dict],
        iou_threshold: float = 0.50,
    ) -> List[Dict]:
        """Merge YOLO+SAM and SAM auto results, deduplicate by mask IoU."""
        all_results = []
        for r in yolo_results:
            r["source"] = "yolo"
            r["composite_score"] = r.get("sam_score", 0)
            r["area_px"] = r.get("area_640", 0)
            all_results.append(r)
        for r in sam_auto_results:
            all_results.append(r)

        if len(all_results) <= 1:
            return sorted(all_results, key=lambda r: r["composite_score"], reverse=True)

        # Deduplicate by mask IoU
        sorted_all = sorted(all_results, key=lambda r: r["composite_score"], reverse=True)
        keep = []
        for r in sorted_all:
            if "mask" not in r or r["mask"] is None:
                keep.append(r)
                continue
            mask_a = r["mask"].astype(bool) if r["mask"].dtype != bool else r["mask"]
            duplicate = False
            for k in keep:
                if "mask" not in k or k["mask"] is None:
                    continue
                mask_b = k["mask"].astype(bool) if k["mask"].dtype != bool else k["mask"]
                intersection = np.logical_and(mask_a, mask_b).sum()
                union = np.logical_or(mask_a, mask_b).sum()
                iou = intersection / max(union, 1)
                if iou > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(r)

        return sorted(keep, key=lambda r: r["composite_score"], reverse=True)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_merged(self, image: np.ndarray, planes: List[Dict], alpha: float = 0.35) -> np.ndarray:
        """Draw merged results with different colors for YOLO vs SAM auto sources."""
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (0, 128, 255),
            (128, 0, 255), (255, 0, 128), (0, 255, 128),
        ]
        overlay = image.copy()
        h, w = image.shape[:2]

        for i, p in enumerate(planes):
            color = colors[i % len(colors)]
            # YOLO-sourced = solid, SAM auto = dashed effect
            source = p.get("source", "yolo")

            # Draw polygon
            contour = p.get("contour", [])
            if len(contour) > 0 and contour.ndim == 3:
                # YOLO contour format: (N, 1, 2)
                poly = contour.reshape(-1, 2)
                if source == "yolo":
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, color, 2)
                else:
                    # SAM auto: show with lighter fill
                    cv2.fillPoly(overlay, [poly], color)
                    cv2.polylines(overlay, [poly], True, (255, 255, 255), 2)

                # Label
                cx, cy = int(np.mean(poly[:, 0])), int(np.mean(poly[:, 1]))
                label = "Y{:d}".format(i) if source == "yolo" else "S{:d}".format(i)
                area = p.get("area_px", 0)
                score = p.get("composite_score", p.get("sam_score", 0))
                text = "{} {:.0f}px".format(label, area)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(overlay, (cx - tw // 2 - 2, cy - th - 2),
                              (cx + tw // 2 + 2, cy + 2), (0, 0, 0), -1)
                cv2.putText(overlay, text, (cx - tw // 2, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return blended

    def unload(self):
        """Free models from memory."""
        if self._segmenter:
            self._segmenter.unload()
            self._segmenter = None
        self._loaded = False
