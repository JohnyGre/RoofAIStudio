#!/usr/bin/env python3
"""
Nástroj na výpočet výšok budov z DMP 1.0 a DMR 5.0 (ZBGIS).

Použitie:
    # 1. Normalizovaný DSM (výšková mapa budov) z dvoch GeoTIFFov
    python lidar_height.py ndsm --dmp dmp_1_0.tif --dmr dmr_5_0.tif --output ndsm.tif

    # 2. Výška budovy na konkrétnych súradniciach
    python lidar_height.py sample --ndsm ndsm.tif --lat 48.1486 --lon 17.1077

    # 3. Profil strechy (výšky pozdĺž línie hrebeň-odkvap)
    python lidar_height.py profile --ndsm ndsm.tif --x1 571234 --y1 5332100 --x2 571250 --y2 5332100

Požiadavky:
    pip install rasterio numpy click pyproj

Zdroje dát:
    DMP 1.0 (Digitálny model povrchu):
        https://zbgis.skgeodesy.sk → Na stiahnutie → LLS / DMP 1.0

    DMR 5.0 (Digitálny model reliéfu):
        https://zbgis.skgeodesy.sk → Na stiahnutie → DMR 5.0

    Surový LiDAR (.LAZ):
        https://zbgis.skgeodesy.sk → Na stiahnutie → LLS → .LAZ súbory
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

try:
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
except ImportError:
    print("Chýba rasterio. Nainštaluj: pip install rasterio")
    sys.exit(1)

try:
    import click
except ImportError:
    print("Chýba click. Nainštaluj: pip install click")
    sys.exit(1)


# ──────────────────────────────────────────────
#  Dátové štruktúry
# ──────────────────────────────────────────────

@dataclass
class RoofProfile:
    """Výškový profil strechy — od odkvapu po hrebeň."""
    distance_m: List[float]
    elevation_m: List[float]
    ground_elevation: float
    min_height: float
    max_height: float
    ridge_position: float
    slope_percent: float
    slope_degrees: float

    def summary(self) -> str:
        return (
            f"Profil strechy:\n"
            f"  Výška hrebeňa:  {self.max_height:.2f} m\n"
            f"  Výška odkvapu:  {self.min_height:.2f} m\n"
            f"  Sklon:          {self.slope_degrees:.1f}° ({self.slope_percent:.0f}%)\n"
            f"  Terén:          {self.ground_elevation:.2f} m n.m."
        )


# ──────────────────────────────────────────────
#  Jadro: Normalizovaný DSM
# ──────────────────────────────────────────────

def compute_ndsm(
    dmp_path: Path,
    dmr_path: Path,
    output_path: Path,
    clip_to_dmr: bool = True,
) -> Tuple[Path, dict]:
    """
    nDSM = DMP (povrch s budovami) − DMR (holý terén)
    Výsledok: terén ≈ 0 m, budovy = reálna výška v metroch.

    Vstup: DMP 1.0 a DMR 5.0 GeoTIFF zo ZBGIS
    Výstup: Normalizovaný DSM GeoTIFF (LZW kompresia, Float32)
    """
    with rasterio.open(dmp_path) as dmp_src, rasterio.open(dmr_path) as dmr_src:
        # Orezať DMP na rozsah DMR
        if clip_to_dmr:
            window = from_bounds(*dmr_src.bounds, dmp_src.transform)
            dmp_data = dmp_src.read(1, window=window, boundless=True)
            dmp_transform = dmp_src.window_transform(window)
        else:
            dmp_data = dmp_src.read(1)
            dmp_transform = dmp_src.transform

        dmr_data = dmr_src.read(1)
        dmr_transform = dmr_src.transform

        # Zarovnanie rozlíšení
        if dmp_data.shape != dmr_data.shape:
            from rasterio.warp import reproject
            import rasterio.enums
            dmr_aligned = np.zeros_like(dmp_data, dtype=np.float32)
            reproject(
                source=dmr_data,
                destination=dmr_aligned,
                src_transform=dmr_transform,
                src_crs=dmr_src.crs,
                dst_transform=dmp_transform,
                dst_crs=dmp_src.crs,
                resampling=rasterio.enums.Resampling.bilinear,
            )
            dmr_data = dmr_aligned

        # nDSM = DMP − DMR
        nodata_mask = (dmp_data <= -999) | (dmr_data <= -999)
        ndsm = dmp_data.astype(np.float32) - dmr_data.astype(np.float32)
        ndsm[nodata_mask] = -9999
        ndsm = np.maximum(ndsm, 0)  # numerický šum → 0

        # Zápis GeoTIFF
        profile = dmp_src.profile.copy()
        profile.update(
            driver="GTiff", dtype=np.float32, count=1,
            nodata=-9999, compress="lzw", transform=dmp_transform,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndsm, 1)

        meta = {
            "shape": ndsm.shape,
            "resolution_m": dmp_transform[0],
            "min_height_m": float(ndsm[~nodata_mask].min()),
            "max_height_m": float(ndsm[~nodata_mask].max()),
            "mean_height_m": float(ndsm[~nodata_mask].mean()),
        }
    return output_path, meta


# ──────────────────────────────────────────────
#  Vzorkovanie výšky
# ──────────────────────────────────────────────

def sample_height_at(
    ndsm_path: Path, x: float, y: float, epsg_in: int = 4326
) -> Optional[float]:
    """Výška budovy [m] na súradnici (WGS84 lat/lon alebo S-JTSK)."""
    with rasterio.open(ndsm_path) as src:
        if epsg_in != src.crs.to_epsg():
            xs, ys = transform(f"EPSG:{epsg_in}", src.crs, [x], [y])
            x, y = xs[0], ys[0]
        row, col = src.index(x, y)
        if row < 0 or col < 0 or row >= src.height or col >= src.width:
            return None
        val = src.read(1)[row, col]
        return float(val) if val > -999 and val != src.nodata else None


def sample_height_latlon(ndsm_path: Path, lat: float, lon: float) -> Optional[float]:
    """Výška budovy z lat/lon (WGS84)."""
    return sample_height_at(ndsm_path, lon, lat, epsg_in=4326)


# ──────────────────────────────────────────────
#  Výškový profil strechy
# ──────────────────────────────────────────────

def roof_profile(
    ndsm_path: Path,
    x1: float, y1: float,
    x2: float, y2: float,
    epsg_in: Optional[int] = None,
    num_samples: int = 50,
) -> RoofProfile:
    """
    Profil pozdĺž línie — typicky odkvap → hrebeň → odkvap.

    Súradnice v rovnakom EPSG ako raster (S-JTSK EPSG:5514),
    alebo uveď epsg_in=4326 pre WGS84.
    """
    with rasterio.open(ndsm_path) as src:
        if epsg_in and epsg_in != src.crs.to_epsg():
            pts = transform(f"EPSG:{epsg_in}", src.crs, [x1, x2], [y1, y2])
            x1, y1 = pts[0][0], pts[1][0]
            x2, y2 = pts[0][1], pts[1][1]

        t = np.linspace(0, 1, num_samples)
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        total_dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        heights, distances = [], []
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            row, col = src.index(xi, yi)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1)[row, col]
                if val > -999 and val != src.nodata:
                    heights.append(float(val))
                    distances.append(float(t[i] * total_dist))

        if not heights:
            raise ValueError("Žiadne platné výšky — bod mimo rastra?")

        ha, da = np.array(heights), np.array(distances)
        ridge_idx = np.argmax(ha)

        # Sklon: lineárna regresia od odkvapu po hrebeň
        if ridge_idx > 0:
            sx = da[:ridge_idx + 1]
            sy = ha[:ridge_idx + 1]
            if len(sx) > 1:
                A = np.vstack([sx, np.ones_like(sx)]).T
                m, _ = np.linalg.lstsq(A, sy, rcond=None)[0]
                slope_pct = abs(m) * 100
                slope_deg = np.degrees(np.arctan(abs(m)))
            else:
                slope_pct = slope_deg = 0.0
        else:
            slope_pct = slope_deg = 0.0

        return RoofProfile(
            distance_m=da.tolist(),
            elevation_m=ha.tolist(),
            ground_elevation=0.0,
            min_height=float(np.min(ha)),
            max_height=float(np.max(ha)),
            ridge_position=da[ridge_idx],
            slope_percent=slope_pct,
            slope_degrees=slope_deg,
        )


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

@click.group()
def cli():
    """RoofAIStudio — výšky budov z LLS/DMP/DMR (ZBGIS)."""
    pass


@cli.command()
@click.option("--dmp", required=True, type=click.Path(exists=True))
@click.option("--dmr", required=True, type=click.Path(exists=True))
@click.option("--output", default="ndsm.tif", type=click.Path())
@click.option("--clip/--no-clip", default=True)
def ndsm(dmp, dmr, output, clip):
    """nDSM = DMP − DMR: výšková mapa budov."""
    out, meta = compute_ndsm(Path(dmp), Path(dmr), Path(output), clip)
    click.echo(f"nDSM: {out}")
    click.echo(f"  Rozlíšenie: {meta['resolution_m']:.2f} m/px")
    click.echo(f"  Výšky: {meta['min_height_m']:.2f} – {meta['max_height_m']:.2f} m "
               f"(priemer {meta['mean_height_m']:.2f} m)")


@cli.command()
@click.option("--ndsm", required=True, type=click.Path(exists=True))
@click.option("--lat", type=float)
@click.option("--lon", type=float)
@click.option("--x", type=float)
@click.option("--y", type=float)
@click.option("--epsg", type=int, default=4326)
def sample(ndsm, lat, lon, x, y, epsg):
    """Výška budovy na súradnici."""
    if lat is not None and lon is not None:
        h = sample_height_latlon(Path(ndsm), lat, lon)
    elif x is not None and y is not None:
        h = sample_height_at(Path(ndsm), x, y, epsg_in=epsg)
    else:
        click.echo("Zadaj --lat/--lon alebo --x/--y", err=True)
        return
    if h is None:
        click.echo("Bod mimo rastra / NoData", err=True)
    else:
        click.echo(f"Výška budovy: {h:.2f} m")


@cli.command()
@click.option("--ndsm", required=True, type=click.Path(exists=True))
@click.option("--x1", required=True, type=float)
@click.option("--y1", required=True, type=float)
@click.option("--x2", required=True, type=float)
@click.option("--y2", required=True, type=float)
@click.option("--epsg", type=int)
@click.option("--csv", type=click.Path())
def profile(ndsm, x1, y1, x2, y2, epsg, csv):
    """Výškový profil strechy (odkvap → hrebeň)."""
    try:
        rp = roof_profile(Path(ndsm), x1, y1, x2, y2, epsg, num_samples=100)
        click.echo(rp.summary())
        if csv:
            out = Path(csv)
            np.savetxt(
                out,
                np.column_stack([rp.distance_m, rp.elevation_m]),
                delimiter=",", header="distance_m,height_m",
                comments="", fmt="%.3f",
            )
            click.echo(f"CSV: {out}")
    except ValueError as e:
        click.echo(f"Chyba: {e}", err=True)


# ═══════════════════════════════════════════════════════════
#  QGIS ekvivalent (referenčný návod)
# ═══════════════════════════════════════════════════════════
QGIS_GUIDE = """
QGIS POSTUP: Normalizovaný DSM (DMP − DMR)
───────────────────────────────────────────
1. Otvor QGIS → Vrstva → Pridať rastrovú vrstvu → DMP_1_0.tif, DMR_5_0.tif

2. Raster Calculator (Raster → Raster Calculator):
     Výraz: "DMP_1_0@1" - "DMR_5_0@1"
     Výstup: ndsm.tif (GeoTIFF, LZW)

3. Stylizácia (Vlastnosti vrstvy → Symbology):
     Typ: Singleband pseudocolor
     Min=0, Max=30, prechod: čierna→zelená→žltá→červená

4. Meranie: Identify Features (Ctrl+Shift+I) → klik na strechu

5. Profil: Pluginy → Profile Tool → línia odkvap–hrebeň → Export CSV

6. LiDAR priamo (.LAZ): QGIS 3.26+
     Data Source Manager → Point Cloud → .LAZ súbor
     Symbology → Attribute by Ramp → Z
     Identify → klik na bod = presná výška
"""


if __name__ == "__main__":
    cli()
