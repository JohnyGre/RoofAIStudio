# -*- coding: utf-8 -*-
"""
pointcloud.py — Načítanie a čistenie LAZ bodového mračna (Fáza C).

Pipeline:
    1. Načítaj LAZ súbory (laspy, S-JTSK JTSK03 + Bpv)
    2. Class filter: nechaj len relevantné triedy (2=terén, 6=budovy, 5=vysoká vegetácia)
    3. Priestorový filter: okruh okolo cieľového bodu (EPSG:5514)
    4. Voxel downsample (zníž hustotu, odstráň duplicity medzi prekrývajúcimi sa LAZ)
    5. SOR (Statistical Outlier Removal) — odstráň riedke odľahlé body

Výstup: numpy array (N, 3) v S-JTSK (EPSG:5514) + Bpv výšky.

POUŽITIE:
    from app.core.pointcloud import load_roof_cloud
    pts = load_roof_cloud(laz_files, center_xy=(-535663.66, -1256478.28), radius=40.0)
"""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import laspy
    HAS_LASPY = True
except ImportError:  # pragma: no cover
    HAS_LASPY = False

# LAS classification kódy (ASPRS)
CLASS_GROUND = 2
CLASS_LOW_VEG = 3
CLASS_MED_VEG = 4
CLASS_HIGH_VEG = 5
CLASS_BUILDING = 6
CLASS_LOW_POINT = 7

# Default: budovy + vegetácia (pre strechu hlavne 6, ale fasády okolo môžu byť 5)
DEFAULT_CLASSES = (CLASS_BUILDING, CLASS_HIGH_VEG, CLASS_MED_VEG, CLASS_LOW_VEG)


def load_laz_files(
    laz_files: Sequence[str],
    classes: Optional[Sequence[int]] = None,
    center_xy: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
) -> np.ndarray:
    """
    Načítaj LAZ súbory a vráť body (N, 3) v S-JTSK.

    Args:
        laz_files: cesty k .laz súborom
        classes: povolené LAS triedy (None = všetky)
        center_xy: (x, y) stredu záujmu v EPSG:5514
        radius: polomer okruhu okolo centra (m)
    """
    if not HAS_LASPY:
        raise RuntimeError("laspy nie je nainštalovaný — `pip install laspy[lazrs]`")

    all_pts = []
    for path in laz_files:
        if not os.path.exists(path):
            print(f"  ⚠️ Chýba: {path}")
            continue
        las = laspy.read(path)
        xs = np.asarray(las.x)
        ys = np.asarray(las.y)
        zs = np.asarray(las.z)

        mask = np.ones(len(xs), dtype=bool)
        if classes is not None:
            cls = np.asarray(las.classification)
            mask &= np.isin(cls, classes)
        if center_xy is not None and radius is not None:
            dist = np.sqrt((xs - center_xy[0]) ** 2 + (ys - center_xy[1]) ** 2)
            mask &= dist <= radius

        if mask.sum() > 0:
            all_pts.append(np.column_stack([xs[mask], ys[mask], zs[mask]]))
            print(f"  {os.path.basename(path)}: {mask.sum()} bodov")

    if not all_pts:
        raise ValueError("Žiadne body nevyhovujú filtrom")

    return np.vstack(all_pts)


def voxel_downsample(pts: np.ndarray, voxel_size: float = 0.3) -> np.ndarray:
    """
    Voxel downsample — jedno reprezentatívne body na voxel.
    Odstráni duplicity medzi prekrývajúcimi sa LAZ súbormi.
    """
    if len(pts) == 0:
        return pts
    grid = np.floor(pts[:, :2] / voxel_size).astype(np.int64)
    # Unikátne voxely — z každého vezmi prvý bod (alebo centroid)
    _, uniq_idx = np.unique(grid, axis=0, return_index=True)
    return pts[uniq_idx]


def statistical_outlier_removal(
    pts: np.ndarray,
    k: int = 20,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    SOR — odstráň body, ktorých priemerná vzdialenosť k k-najbližším
    susedom presahuje mean + std_ratio * std.
    """
    if len(pts) < k + 1:
        return pts
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return pts

    tree = cKDTree(pts)
    # Vzdialenosti ku k najbližším susedom (bez seba)
    dist, _ = tree.query(pts, k=k + 1)
    mean_dist = dist[:, 1:].mean(axis=1)

    mu = mean_dist.mean()
    sigma = mean_dist.std()
    if sigma < 1e-9:
        return pts
    keep = mean_dist <= mu + std_ratio * sigma
    return pts[keep]


def classify_roof_points(
    pts: np.ndarray,
    ground_z: Optional[float] = None,
    min_height: float = 2.0,
) -> np.ndarray:
    """
    Vyber body ktoré sú NAD terénom (potenciálna strecha).
    Ak ground_z nie je dané, použije sa 2. percentil Z (terén).
    """
    if ground_z is None:
        ground_z = float(np.percentile(pts[:, 2], 2))
    return pts[pts[:, 2] >= ground_z + min_height]


def load_roof_cloud(
    laz_dir: str,
    center_xy: Tuple[float, float],
    radius: float = 40.0,
    classes: Sequence[int] = DEFAULT_CLASSES,
    voxel: float = 0.3,
    sor: bool = True,
    min_height: float = 2.0,
) -> np.ndarray:
    """
    Kompletný load LAZ → čisté strešné body.

    Returns:
        (N, 3) array — body strechy v EPSG:5514 + Bpv
    """
    laz_files = sorted(
        os.path.join(laz_dir, f)
        for f in os.listdir(laz_dir)
        if f.endswith(".laz")
    )
    if not laz_files:
        raise FileNotFoundError(f"Žiadne .laz súbory v {laz_dir}")

    print(f"Načítavam {len(laz_files)} LAZ súborov...")
    pts = load_laz_files(laz_files, classes=classes, center_xy=center_xy, radius=radius)
    print(f"Po class+priestor filtri: {len(pts)} bodov")

    pts = voxel_downsample(pts, voxel)
    print(f"Po voxel downsample ({voxel}m): {len(pts)} bodov")

    if sor:
        pts = statistical_outlier_removal(pts)
        print(f"Po SOR: {len(pts)} bodov")

    roof = classify_roof_points(pts, min_height=min_height)
    print(f"Strešné body (nad {min_height}m nad terénom): {len(roof)}")

    return roof
