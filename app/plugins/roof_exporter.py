# -*- coding: utf-8 -*-
"""
Export roof planes to colored OBJ/PLY for GUI viewer.

Each plane = separate group with:
- Filled faces (semi-transparent)
- Edge lines colored by type: h=hreben(blue), n=narozie(green), u=uzlabie(red), o=okap(yellow), s=stit(purple)
- Plane label as vertex color / group name
"""
import os
import sys
import math
from typing import List, Dict

import numpy as np

# Edge type -> RGB color (0-255)
EDGE_COLORS = {
    "h": (0, 100, 255),     # hreben - modrá
    "n": (0, 200, 0),       # narozie - zelená
    "u": (255, 50, 50),     # úžľabie - červená
    "o": (255, 200, 0),     # okap - žltá
    "s": (180, 0, 180),     # štít - fialová
}

# Plane colors for fill (pastel, semi-transparent feel)
PLANE_COLORS = [
    (100, 180, 255),   # light blue
    (255, 180, 100),   # orange
    (100, 255, 150),   # green
    (255, 130, 130),   # pink
    (200, 160, 255),   # lavender
    (160, 255, 200),   # mint
    (255, 220, 150),   # peach
    (150, 220, 255),   # sky blue
    (200, 200, 200),   # gray (flat roof)
]


def export_roof_planes_obj(planes: List[Dict], output_path: str) -> str:
    """
    Export roof planes as colored OBJ file.

    Args:
        planes: output from mesh_to_roof_planes_exact()
        output_path: where to save .obj

    Returns:
        output_path
    """
    good_planes = [p for p in planes if not p.get("low_confidence") and p.get("vertices")]

    if not good_planes:
        raise ValueError("No valid planes with vertices to export")

    lines = []
    vertex_offset = 0

    for pi, plane in enumerate(good_planes):
        pid = plane.get("id", f"R{pi+1}")
        ptype = plane.get("type", "?")
        area = plane.get("area_m2", 0)
        pitch = plane.get("pitch_deg", 0)

        # Group header
        lines.append(f"g {pid}_{ptype}")
        lines.append(f"# Area={area:.1f}m2 Pitch={pitch:.1f}deg")

        verts = plane["vertices"]
        n_verts = len(verts)

        # Write vertices
        for v in verts:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        # Write filled face (all vertices as one polygon)
        if n_verts >= 3:
            face_indices = " ".join(str(i + 1 + vertex_offset) for i in range(n_verts))
            lines.append(f"f {face_indices}")

        # Write edge lines colored by type
        edges = plane.get("edges", [])
        for ei, edge in enumerate(edges):
            etype = edge.get("type", "o")
            eid = edge.get("id", f"e{ei}")
            start = edge.get("start", [0, 0, 0])
            end = edge.get("end", [0, 0, 0])
            length = edge.get("length_m", 0)

            r, g, b = EDGE_COLORS.get(etype, (128, 128, 128))

            # Edge vertices (as line segments)
            lines.append(f"# edge {eid} type={etype} len={length:.2f}m")
            lines.append(f"v {start[0]:.4f} {start[1]:.4f} {start[2]:.4f}")
            lines.append(f"v {end[0]:.4f} {end[1]:.4f} {end[2]:.4f}")

            e_v1 = len([l for l in lines if l.startswith("v ")]) - 1
            e_v2 = e_v1 + 1
            lines.append(f"l {e_v1} {e_v2}")

        vertex_offset += n_verts + sum(2 for e in edges)  # +2 per edge for line verts

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    return output_path


def export_roof_planes_ply(planes: List[Dict], output_path: str) -> str:
    """
    Export roof planes as PLY with vertex colors and edges.

    PLY format supports per-vertex colors (RGBA) which is great for
    distinguishing planes and edge types in MeshLab/Blender.
    """
    good_planes = [p for p in planes if not p.get("low_confidence") and p.get("vertices")]

    all_verts = []
    all_faces = []
    all_edges = []  # (v1_idx, v2_idx, r, g, b)

    v_offset = 0

    for pi, plane in enumerate(good_planes):
        verts = plane["vertices"]
        pc = PLANE_COLORS[pi % len(PLANE_COLORS)]

        # Face vertices with plane color
        for v in verts:
            all_verts.append((v[0], v[1], v[2], pc[0], pc[1], pc[2], 180))  # RGBA, alpha=180

        n = len(verts)
        if n >= 3:
            all_faces.append(tuple(range(v_offset, v_offset + n)))
        v_offset += n

        # Edge lines
        edges = plane.get("edges", [])
        for edge in edges:
            etype = edge.get("type", "o")
            ec = EDGE_COLORS.get(etype, (128, 128, 128))
            start = edge.get("start", [0, 0, 0])
            end = edge.get("end", [0, 0, 0])

            i1 = len(all_verts)
            all_verts.append((start[0], start[1], start[2], ec[0], ec[1], ec[2], 255))
            i2 = len(all_verts)
            all_verts.append((end[0], end[1], end[2], ec[0], ec[1], ec[2], 255))
            all_edges.append((i1, i2))

    # Write PLY
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write(f"element edge {len(all_edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\n")
        f.write("end_header\n")

        for v in all_verts:
            f.write(f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {int(v[3])} {int(v[4])} {int(v[5])} {int(v[6])}\n")

        for face in all_faces:
            idx_str = " ".join(str(i) for i in face)
            f.write(f"{len(face)} {idx_str}\n")

        for e in all_edges:
            f.write(f"{e[0]} {e[1]}\n")

    return output_path


