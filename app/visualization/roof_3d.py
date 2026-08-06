"""
Roof3DExporter — Export roof plane data to 3D visualizations.

Supports:
  - export_to_html(json_path, output_path) → standalone HTML with Three.js
  - export_to_gltf(json_path, output_path) → glTF 2.0 3D model
"""

import json
import os
import struct
import base64
from typing import Any


class Roof3DExporter:
    """Exports roof plane data to HTML (Three.js) or glTF 2.0."""

    COLOR_MAP = {
        "Zelená": "#4CAF50",
        "Modrá": "#2196F3",
        "Červená": "#F44336",
        "Akvamarínová": "#00BCD4",
        "Fialová": "#9C27B0",
        "Žltá": "#FFEB3B",
    }

    @staticmethod
    def _load_json(json_path: str) -> dict:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _extract_vertices(contour: list, px_per_m: float) -> list:
        """Convert OpenCV contour [[[x,y]], ...] to list of (x_m, z_m)."""
        pts = []
        for item in contour:
            if isinstance(item, list) and len(item) >= 1:
                inner = item[0] if isinstance(item[0], list) else item
                x_px, y_px = inner[0], inner[1]
                pts.append((x_px / px_per_m, y_px / px_per_m))
        # Deduplicate consecutive points
        deduped = []
        for pt in pts:
            if not deduped or (abs(pt[0] - deduped[-1][0]) > 0.001 or abs(pt[1] - deduped[-1][1]) > 0.001):
                deduped.append(pt)
        # Remove last if same as first
        if len(deduped) >= 3:
            if abs(deduped[0][0] - deduped[-1][0]) < 0.001 and abs(deduped[0][1] - deduped[-1][1]) < 0.001:
                deduped.pop()
        return deduped

    @staticmethod
    def _compute_height(class_name: str, area_m2: float, all_planes: list) -> float:
        """Compute roof height based on class and area."""
        if class_name == "slope_flat":
            return 0.3
        # Scale between 2m and 5m based on area
        non_flat = [p for p in all_planes if p.get("class_name") != "slope_flat"]
        if not non_flat:
            return 2.5
        min_area = min(p["area_m2"] for p in non_flat)
        max_area = max(p["area_m2"] for p in non_flat)
        if max_area == min_area:
            return 3.5
        return 2.0 + (area_m2 - min_area) / (max_area - min_area) * 3.0

    # ── HTML Export ────────────────────────────────────────────

    @classmethod
    def export_to_html(cls, json_path: str, output_path: str) -> str:
        """Generate a standalone HTML file with embedded Three.js viewer."""
        data = cls._load_json(json_path)
        html = cls._build_html(data)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

    @classmethod
    def _build_html(cls, data: dict) -> str:
        planes_json = json.dumps(data, ensure_ascii=False, indent=2)
        return _HTML_TEMPLATE.replace("/* __PLANES_DATA_PLACEHOLDER__ */", planes_json)

    # ── glTF Export ────────────────────────────────────────────

    @classmethod
    def export_to_gltf(cls, json_path: str, output_path: str) -> str:
        """Generate a glTF 2.0 binary file (.glb) from roof data."""
        data = cls._load_json(json_path)
        px_per_m = data["px_per_m"]
        planes = data["planes"]

        # Collect all vertices and indices
        all_positions = []
        all_normals = []
        all_indices = []
        all_colors = []
        vertex_offset = 0

        for plane in planes:
            verts_2d = cls._extract_vertices(plane["contour"], px_per_m)
            if len(verts_2d) < 3:
                continue
            height = cls._compute_height(plane["class_name"], plane["area_m2"], planes)
            color_hex = cls.COLOR_MAP.get(plane.get("color_name", ""), "#888888")
            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0

            # Bottom face vertices
            for x, z in verts_2d:
                all_positions.extend([x, 0, z])
                all_normals.extend([0, -1, 0])
                all_colors.extend([r, g, b, 0.7])
            # Top face vertices
            for x, z in verts_2d:
                all_positions.extend([x, height, z])
                all_normals.extend([0, 1, 0])
                all_colors.extend([r, g, b, 0.7])

            n = len(verts_2d)
            # Triangulate bottom + top faces (fan)
            for i in range(1, n - 1):
                all_indices.extend([vertex_offset, vertex_offset + i, vertex_offset + i + 1])
                all_indices.extend([vertex_offset + n, vertex_offset + n + i + 1, vertex_offset + n + i])
            # Side walls
            for i in range(n):
                j = (i + 1) % n
                a = vertex_offset + i
                b = vertex_offset + j
                c = vertex_offset + n + i
                d = vertex_offset + n + j
                all_indices.extend([a, b, d, a, d, c])

            vertex_offset += 2 * n

        # Write glTF binary
        glb = _build_glb(all_positions, all_normals, all_colors, all_indices)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(glb)
        return output_path


def _build_glb(positions, normals, colors, indices):
    """Build a minimal glTF 2.0 GLB binary."""
    import struct as _st

    pos_bytes = _st.pack(f"<{len(positions)}f", *positions)
    nrm_bytes = _st.pack(f"<{len(normals)}f", *normals)
    col_bytes = _st.pack(f"<{len(colors)}f", *colors)
    idx_bytes = _st.pack(f"<{len(indices)}I", *indices)

    # Pad to 4-byte alignment
    def pad(b):
        while len(b) % 4 != 0:
            b += b"\x00"
        return b

    pos_bytes = pad(pos_bytes)
    nrm_bytes = pad(nrm_bytes)
    col_bytes = pad(col_bytes)
    idx_bytes = pad(idx_bytes)

    buf_data = b"".join([pos_bytes, nrm_bytes, col_bytes, idx_bytes])
    buf_length = len(buf_data)

    vert_count = len(positions) // 3
    idx_count = len(indices)

    gltf = {
        "asset": {"version": "2.0", "generator": "RoofAIStudio Roof3DExporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "COLOR_0": 2
                },
                "indices": 3,
                "mode": 4
            }]
        }],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": vert_count, "type": "VEC3", "min": [min(positions[i::3]) for i in range(3)], "max": [max(positions[i::3]) for i in range(3)]},
            {"bufferView": 1, "componentType": 5126, "count": vert_count, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": vert_count, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5125, "count": idx_count, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(nrm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(nrm_bytes), "byteLength": len(col_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(nrm_bytes) + len(col_bytes), "byteLength": len(idx_bytes), "target": 34963},
        ],
        "buffers": [{"byteLength": buf_length}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.8, 1.0]}, "alphaMode": "BLEND", "doubleSided": True}],
    }

    import json as _json
    json_str = _json.dumps(gltf, separators=(",", ":"))
    json_bytes = json_str.encode("utf-8")
    json_bytes = pad(json_bytes)

    # GLB header
    total_len = 12 + 8 + len(json_bytes) + 8 + len(buf_data)
    header = _st.pack("<I", 0x46546C67) + _st.pack("<I", 2) + _st.pack("<I", total_len)
    chunk0 = _st.pack("<I", len(json_bytes)) + _st.pack("<I", 0x4E4F534A) + json_bytes
    chunk1 = _st.pack("<I", len(buf_data)) + _st.pack("<I", 0x004E4942) + buf_data

    return header + chunk0 + chunk1


# ── HTML Template ─────────────────────────────────────────────

_HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RoofAIStudio — 3D Strecha</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; overflow: hidden; background: #1a1a2e; }
  #canvas-container { width: 100vw; height: 100vh; position: relative; }
  canvas { display: block; }

  /* Tooltip */
  #tooltip {
    position: absolute; display: none; background: rgba(20,20,40,0.92); color: #fff;
    padding: 10px 14px; border-radius: 8px; font-size: 13px; line-height: 1.5;
    pointer-events: none; border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(8px); max-width: 260px; z-index: 100;
  }
  #tooltip .tt-title { font-weight: 700; font-size: 14px; margin-bottom: 4px; }
  #tooltip .tt-row { display: flex; justify-content: space-between; gap: 12px; }
  #tooltip .tt-label { color: #aaa; }
  #tooltip .tt-val { text-align: right; font-weight: 500; }

  /* Legend */
  #legend {
    position: absolute; bottom: 20px; left: 20px; background: rgba(20,20,40,0.88);
    color: #ddd; padding: 12px 16px; border-radius: 10px; font-size: 12px;
    border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(6px);
    z-index: 50; user-select: none; line-height: 1.6;
  }
  #legend h3 { margin: 0 0 8px 0; font-size: 13px; color: #fff; }
  #legend .leg-row { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
  #legend .leg-swatch { width: 16px; height: 16px; border-radius: 3px; flex-shrink: 0; }

  /* Info bar */
  #info-bar {
    position: absolute; top: 16px; left: 50%; transform: translateX(-50%);
    background: rgba(20,20,40,0.85); color: #ccc; padding: 6px 18px;
    border-radius: 20px; font-size: 12px; border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(6px); z-index: 50; pointer-events: none;
  }

  /* Edit mode indicator */
  #edit-indicator {
    position: absolute; top: 60px; left: 50%; transform: translateX(-50%);
    background: #e74c3c; color: #fff; padding: 5px 16px; border-radius: 14px;
    font-size: 12px; font-weight: 600; z-index: 50; display: none;
  }
  #edit-indicator.active { display: block; }

  /* Help hint */
  #help-hint {
    position: absolute; bottom: 20px; right: 20px; color: rgba(255,255,255,0.35);
    font-size: 11px; z-index: 50; pointer-events: none;
  }
</style>
</head>
<body>
<div id="canvas-container">
  <div id="info-bar">🏠 Strecha — načítava sa…</div>
  <div id="edit-indicator">✏️ Edit mód — ťahaj vrcholy | Delete = zmazať | Dvojklik = ukončiť</div>
  <div id="legend"></div>
  <div id="tooltip"></div>
  <div id="help-hint">🖱️ Ľavý+ťah = rotácia | Koliesko = zoom | Pravý+ťah = posun | Klik = info | Dvojklik = edit</div>
</div>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.168.0/examples/jsm/"
  }
}
</script>

<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── Embed data ────────────────────────────────────────────────
const ROOF_DATA = /* __PLANES_DATA_PLACEHOLDER__ */;

// ── Color helpers ─────────────────────────────────────────────
const COLOR_MAP = {
  "Zelená": 0x4CAF50, "Modrá": 0x2196F3, "Červená": 0xF44336,
  "Akvamarínová": 0x00BCD4, "Fialová": 0x9C27B0, "Žltá": 0xFFEB3B,
};
const DEFAULT_COLOR = 0x888888;

function resolveColor(name) {
  for (const [k, v] of Object.entries(COLOR_MAP)) {
    if (name.includes(k) || k.includes(name)) return v;
  }
  return DEFAULT_COLOR;
}

function hexToCSS(hex) {
  return "#" + hex.toString(16).padStart(6, "0");
}

// ── Scene setup ───────────────────────────────────────────────
const container = document.getElementById("canvas-container");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();

// Sky gradient via background
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 30, 100);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.5, 200);
camera.position.set(30, 22, 35);
camera.lookAt(25, 3, 28);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(25, 3, 28);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 5;
controls.maxDistance = 80;
controls.maxPolarAngle = Math.PI * 0.55;
controls.update();

// ── Lighting ──────────────────────────────────────────────────
const ambient = new THREE.AmbientLight(0x8899cc, 1.6);
scene.add(ambient);

const sun = new THREE.DirectionalLight(0xfff5e8, 3.5);
sun.position.set(40, 50, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 0.5;
sun.shadow.camera.far = 150;
sun.shadow.camera.left = -60;
sun.shadow.camera.right = 60;
sun.shadow.camera.top = 60;
sun.shadow.camera.bottom = -60;
sun.shadow.bias = -0.0005;
scene.add(sun);

const fill = new THREE.DirectionalLight(0x8899cc, 0.8);
fill.position.set(-10, 5, -10);
scene.add(fill);

// ── Ground plane ──────────────────────────────────────────────
const groundGeo = new THREE.PlaneGeometry(120, 120);
const groundMat = new THREE.MeshStandardMaterial({ color: 0x2a2a3e, roughness: 0.9 });
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI / 2;
ground.position.set(25, -0.05, 28);
ground.receiveShadow = true;
scene.add(ground);

// Grid helper
const grid = new THREE.GridHelper(60, 40, 0x444466, 0x222244);
grid.position.set(25, 0.001, 28);
scene.add(grid);

// ── Build roof planes ─────────────────────────────────────────
const pxPerM = ROOF_DATA.px_per_m || 12.1;
const planes = ROOF_DATA.planes || [];
const planeObjects = [];   // { mesh, wire, vertices3D, planeData, group }
const planesMeta = [];       // intermediate: { v2d, colorHex, height, planeData }
const allVertices3D = [];  // for snapping

// Compute height range
const nonFlat = planes.filter(p => p.class_name !== "slope_flat");
const areas = nonFlat.map(p => p.area_m2);
const minArea = Math.min(...areas);
const maxArea = Math.max(...areas);

function computeHeight(className, areaM2) {
  if (className === "slope_flat") return 0.3;
  if (maxArea === minArea) return 3.5;
  return 2.0 + ((areaM2 - minArea) / (maxArea - minArea)) * 3.0;
}

function extractVertices(contour) {
  const pts = [];
  for (const item of contour) {
    const inner = Array.isArray(item[0]) ? item[0] : item;
    pts.push([inner[0] / pxPerM, inner[1] / pxPerM]);
  }
  // Deduplicate consecutive
  const deduped = [];
  for (const pt of pts) {
    if (deduped.length === 0 ||
        Math.abs(pt[0] - deduped[deduped.length - 1][0]) > 0.001 ||
        Math.abs(pt[1] - deduped[deduped.length - 1][1]) > 0.001) {
      deduped.push(pt);
    }
  }
  // Remove last if closed
  if (deduped.length >= 3 &&
      Math.abs(deduped[0][0] - deduped[deduped.length - 1][0]) < 0.001 &&
      Math.abs(deduped[0][1] - deduped[deduped.length - 1][1]) < 0.001) {
    deduped.pop();
  }
  return deduped;
}

function createExtrudedPolygon(v2d, height, colorHex) {
  const group = new THREE.Group();
  const n = v2d.length;
  if (n < 3) return { group, vertices3D: [] };

  // Shape for top/bottom
  const shape = new THREE.Shape();
  shape.moveTo(v2d[0][0], v2d[0][1]);
  for (let i = 1; i < n; i++) shape.lineTo(v2d[i][0], v2d[i][1]);
  shape.closePath();

  const extrudeSettings = { steps: 1, depth: height, bevelEnabled: false };
  const geo = new THREE.ExtrudeGeometry(shape, extrudeSettings);

  // Translate so the base sits at y=0
  // ExtrudeGeometry extrudes along +Z, so we rotate and translate
  // vertices are in XY plane, we want them in XZ plane with Y as height
  geo.rotateX(-Math.PI / 2);  // XY → XZ, extrusion goes up (Y)

  const mat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: 0.55,
    metalness: 0.05,
    transparent: true,
    opacity: 0.72,
    side: THREE.DoubleSide,
    depthWrite: true,
  });

  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = "roof-plane";
  group.add(mesh);

  // Wireframe
  const wireGeo = new THREE.EdgesGeometry(geo, 30);
  const wireMat = new THREE.LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.18 });
  const wire = new THREE.LineSegments(wireGeo, wireMat);
  wire.visible = true;
  group.add(wire);

  // Store 3D vertices for editing: top surface (y = height)
  const vertices3D = v2d.map(([x, z]) => new THREE.Vector3(x, height, z));

  return { group, mesh, wire, vertices3D };
}

planes.forEach(plane => {
  const v2d = extractVertices(plane.contour);
  if (v2d.length < 3) return;
  const colorHex = resolveColor(plane.color_name || "");
  const height = plane.height || computeHeight(plane.class_name, plane.area_m2);

  // Store plane data for connected geometry build
  planesMeta.push({ v2d, colorHex, height, planeData: plane });
});

// ============================================================
// BUILD CONNECTED ROOF GEOMETRY
// Instead of extruding each plane separately, we:
// 1. Collect all unique vertices across all planes
// 2. Determine which vertices are on the outer perimeter
// 3. Assign heights: perimeter=0, internal=ridge_height
// 4. Build top surface from all plane faces
// 5. Build wall quads from perimeter edges down to y=0
// 6. Build bottom face at y=0
// ============================================================

// Step 1+2: Collect unique vertices and count edge ownership
const all2D = [];           // [[x, z], ...]
const vertKeyMap = new Map(); // "x,z" -> index
function getVIdx(x, z) {
  const k = x.toFixed(2) + "," + z.toFixed(2);
  if (!vertKeyMap.has(k)) {
    vertKeyMap.set(k, all2D.length);
    all2D.push([x, z]);
  }
  return vertKeyMap.get(k);
}

const edgeOwners = new Map(); // "a,b" -> count
function edgeKey(a, b) { return a < b ? a + "," + b : b + "," + a; }

const planeFaces = []; // [{indices: [vIdx,...], colorHex, height}]
planesMeta.forEach(pm => {
  const indices = pm.v2d.map(([x, z]) => getVIdx(x, z));
  // Deduplicate consecutive same-index (closed loop fix)
  const deduped = [indices[0]];
  for (let i = 1; i < indices.length; i++) {
    if (indices[i] !== deduped[deduped.length - 1]) deduped.push(indices[i]);
  }
  if (deduped.length >= 3 && deduped[0] === deduped[deduped.length - 1]) deduped.pop();
  if (deduped.length < 3) return;

  for (let i = 0; i < deduped.length; i++) {
    const j = (i + 1) % deduped.length;
    const ek = edgeKey(deduped[i], deduped[j]);
    edgeOwners.set(ek, (edgeOwners.get(ek) || 0) + 1);
  }
  planeFaces.push({ indices: deduped, colorHex: pm.colorHex, height: pm.height, planeData: pm.planeData });
});

// Step 3: Assign 3D heights
const isPerimeter = new Array(all2D.length).fill(false);
for (const [ek, count] of edgeOwners) {
  if (count === 1) {
    const [a, b] = ek.split(",").map(Number);
    isPerimeter[a] = true;
    isPerimeter[b] = true;
  }
}

const RIDGE_H = ROOF_DATA.ridge_height_m || 2.4;
const all3D = all2D.map(([x, z], i) => {
  const y = isPerimeter[i] ? 0.05 : RIDGE_H;
  return new THREE.Vector3(x, y, z);
});

// Step 4: Build top surface (all roof plane faces, triangulated)
const topGroup = new THREE.Group();
planeFaces.forEach(pf => {
  const { indices, colorHex } = pf;
  if (indices.length < 3) return;

  // Create BufferGeometry for this plane's top face
  const verts = indices.map(i => all3D[i]);
  const geo = new THREE.BufferGeometry();
  const positions = [];
  const normals_arr = [];

  // Fan triangulation
  for (let i = 1; i < verts.length - 1; i++) {
    positions.push(
      verts[0].x, verts[0].y, verts[0].z,
      verts[i].x, verts[i].y, verts[i].z,
      verts[i+1].x, verts[i+1].y, verts[i+1].z,
    );
    // Upward normals
    for (let k = 0; k < 3; k++) normals_arr.push(0, 1, 0);
  }

  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals_arr, 3));
  geo.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: 0.5,
    metalness: 0.05,
    side: THREE.DoubleSide,
    depthWrite: true,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.name = "roof-top";
  topGroup.add(mesh);

  // Store for interaction
  planeObjects.push({
    group: topGroup, mesh, wire: null, vertices3D: verts,
    planeData: pf.planeData,
  });
  allVertices3D.push(...verts);
});

scene.add(topGroup);

// Step 5: Build walls from perimeter edges
// Collect perimeter edges as ordered loops
const perimeterEdges = [];
for (const [ek, count] of edgeOwners) {
  if (count === 1) {
    const [a, b] = ek.split(",").map(Number);
    perimeterEdges.push([a, b]);
  }
}

// Chain perimeter edges into closed loop(s)
const usedVerts = new Set();
const loops = [];
while (perimeterEdges.length > 0) {
  // Find a starting edge
  const startEdge = perimeterEdges.shift();
  const loop = [startEdge[0], startEdge[1]];
  usedVerts.add(startEdge[0]);
  usedVerts.add(startEdge[1]);
  let current = startEdge[1];

  let extended = true;
  while (extended) {
    extended = false;
    for (let i = perimeterEdges.length - 1; i >= 0; i--) {
      const [a, b] = perimeterEdges[i];
      if (a === current && !usedVerts.has(b)) {
        loop.push(b);
        usedVerts.add(b);
        current = b;
        perimeterEdges.splice(i, 1);
        extended = true;
        break;
      } else if (b === current && !usedVerts.has(a)) {
        loop.push(a);
        usedVerts.add(a);
        current = a;
        perimeterEdges.splice(i, 1);
        extended = true;
        break;
      }
    }
  }
  if (loop.length >= 3) loops.push(loop);
}

// Build wall quads for each perimeter loop
loops.forEach(loop => {
  const wallGeo = new THREE.BufferGeometry();
  const wPositions = [];
  const wNormals = [];

  for (let i = 0; i < loop.length; i++) {
    const j = (i + 1) % loop.length;
    const a = all3D[loop[i]];
    const b = all3D[loop[j]];

    // Wall quad: a(top), b(top), b(bottom), a(top), b(bottom), a(bottom)
    const aBot = new THREE.Vector3(a.x, 0, a.z);
    const bBot = new THREE.Vector3(b.x, 0, b.z);

    wPositions.push(a.x, a.y, a.z, b.x, b.y, b.z, bBot.x, 0, bBot.z);
    wPositions.push(a.x, a.y, a.z, bBot.x, 0, bBot.z, aBot.x, 0, aBot.z);

    // Wall normal (pointing outward)
    const dx = b.z - a.z;
    const dz = -(b.x - a.x);
    const len = Math.sqrt(dx*dx + dz*dz) || 1;
    const nx = dx / len, nz = dz / len;
    for (let k = 0; k < 6; k++) wNormals.push(nx, 0, nz);
  }

  wallGeo.setAttribute("position", new THREE.Float32BufferAttribute(wPositions, 3));
  wallGeo.setAttribute("normal", new THREE.Float32BufferAttribute(wNormals, 3));
  wallGeo.computeVertexNormals();

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0xd4c5a9,
    roughness: 0.7,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  const walls = new THREE.Mesh(wallGeo, wallMat);
  walls.castShadow = true;
  walls.receiveShadow = true;
  walls.name = "roof-walls";
  scene.add(walls);
});

// Step 6: Bottom face at y=0 using largest perimeter loop
if (loops.length > 0) {
  const largestLoop = loops.reduce((a, b) => a.length >= b.length ? a : b);
  const shape = new THREE.Shape();
  const first = all3D[largestLoop[0]];
  shape.moveTo(first.x, first.z);
  for (let i = 1; i < largestLoop.length; i++) {
    const p = all3D[largestLoop[i]];
    shape.lineTo(p.x, p.z);
  }
  shape.closePath();
  const bottomGeo = new THREE.ShapeGeometry(shape);
  bottomGeo.rotateX(-Math.PI / 2);
  bottomGeo.translate(0, 0.01, 0);
  const bottomMat = new THREE.MeshStandardMaterial({ color: 0x3a3a4a, roughness: 0.9, side: THREE.DoubleSide });
  const bottom = new THREE.Mesh(bottomGeo, bottomMat);
  bottom.receiveShadow = true;
  bottom.name = "roof-bottom";
  scene.add(bottom);
}

// // Center scene
if (planeObjects.length > 0) {
  const box = new THREE.Box3();
  planeObjects.forEach(po => box.expandByObject(po.group));
  const center = new THREE.Vector3();
  box.getCenter(center);
  controls.target.copy(center);
  camera.position.set(center.x + 28, center.y + 22, center.z + 32);
  camera.lookAt(center);
  controls.update();
  document.getElementById("info-bar").textContent =
    `🏠 ${ROOF_DATA.address || "Strecha"} — ${planes.length} rovín`;
}

// ── Legend ────────────────────────────────────────────────────
function buildLegend() {
  const legend = document.getElementById("legend");
  const typeLabels = {
    slope_poly: "Šikmá plocha (polygón)",
    slope_tri: "Šikmá plocha (trojuholník)",
    slope_min: "Minimálny sklon",
    slope_flat: "Plochá strecha",
  };
  const seenColors = new Set();
  let html = "<h3>📐 Legenda</h3>";
  for (const po of planeObjects) {
    const d = po.group.userData;
    const key = d.color_name + "|" + d.class_name;
    if (seenColors.has(key)) continue;
    seenColors.add(key);
    const css = hexToCSS(d.colorHex);
    html += `<div class="leg-row"><span class="leg-swatch" style="background:${css}"></span>${d.color_name} — ${typeLabels[d.class_name] || d.class_name}</div>`;
  }
  legend.innerHTML = html;
}
buildLegend();

// ── Interaction state ─────────────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Line = { threshold: 0.3 };
const mouse = new THREE.Vector2();

let hoveredPlane = null;
let selectedPlane = null;
let editMode = false;
let editPlane = null;
let draggingVertex = null;
let vertexSpheres = [];
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const intersection = new THREE.Vector3();
const offset = new THREE.Vector3();

const tooltip = document.getElementById("tooltip");
const editIndicator = document.getElementById("edit-indicator");

// ── Highlight helpers ─────────────────────────────────────────
function resetAllHighlight() {
  planeObjects.forEach(po => {
    po.mesh.material.emissive?.setHex(0x000000);
    po.wire.material.color.set(0x000000);
    po.wire.material.opacity = 0.18;
  });
}

function highlightPlane(po, color = 0xffff88) {
  po.mesh.material.emissive = new THREE.Color(color);
  po.mesh.material.emissiveIntensity = 0.4;
  po.wire.material.color.set(color);
  po.wire.material.opacity = 0.6;
}

// ── Vertex spheres for edit mode ──────────────────────────────
function createVertexSpheres(vertices3D) {
  removeVertexSpheres();
  const sphereGeo = new THREE.SphereGeometry(0.25, 16, 12);
  vertices3D.forEach((v, i) => {
    const mat = new THREE.MeshStandardMaterial({ color: 0xff6600, roughness: 0.3, emissive: 0x331100 });
    const sphere = new THREE.Mesh(sphereGeo, mat);
    sphere.position.copy(v);
    sphere.userData = { vertexIndex: i, isVertexHandle: true };
    sphere.renderOrder = 999;
    sphere.material.depthTest = false;
    sphere.material.depthWrite = false;
    scene.add(sphere);
    vertexSpheres.push(sphere);
  });
}

function removeVertexSpheres() {
  vertexSpheres.forEach(s => scene.remove(s));
  vertexSpheres = [];
}

function updateVertexSpheres(vertices3D) {
  vertexSpheres.forEach((s, i) => {
    if (i < vertices3D.length) s.position.copy(vertices3D[i]);
  });
}

// ── Rebuild plane geometry ────────────────────────────────────
function rebuildPlane(po) {
  // Remove old group
  scene.remove(po.group);
  po.group.traverse(c => {
    if (c.geometry) c.geometry.dispose();
    if (c.material) {
      if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
      else c.material.dispose();
    }
  });

  // Extract new 2D vertices
  const v2d = po.vertices3D.map(v => [v.x, v.z]);
  const height = po.group.userData.height;
  const colorHex = po.group.userData.colorHex;
  const { group, mesh, wire, vertices3D } = createExtrudedPolygon(v2d, height, colorHex);
  group.userData = po.group.userData;

  scene.add(group);
  po.group = group;
  po.mesh = mesh;
  po.wire = wire;
  po.vertices3D = vertices3D;
}

// ── Raycasting ────────────────────────────────────────────────
function getIntersections(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const meshes = planeObjects.map(po => po.mesh);
  const intersects = raycaster.intersectObjects(meshes, false);
  return intersects;
}

function getVertexIntersection(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  return raycaster.intersectObjects(vertexSpheres, false);
}

function getPlaneByMesh(mesh) {
  return planeObjects.find(po => po.mesh === mesh);
}

// ── Tooltip ───────────────────────────────────────────────────
function showTooltip(po, event) {
  const d = po.group.userData;
  const classLabels = {
    slope_poly: "Šikmá (polygón)", slope_tri: "Šikmá (trojuholník)",
    slope_min: "Minimálny sklon", slope_flat: "Plochá",
  };
  const edges = (d.edge_details || []).map((e, i) =>
    `${i + 1}. ${e.length_m?.toFixed(2) || "?"} m`).join("<br>") || "—";

  tooltip.innerHTML = `
    <div class="tt-title" style="color:${hexToCSS(d.colorHex)}">${d.color_name} — ${classLabels[d.class_name] || d.class_name}</div>
    <div class="tt-row"><span class="tt-label">Plocha:</span><span class="tt-val">${d.area_m2?.toFixed(2)} m²</span></div>
    <div class="tt-row"><span class="tt-label">Obvod:</span><span class="tt-val">${d.perimeter_m?.toFixed(2)} m</span></div>
    <div class="tt-row"><span class="tt-label">Výška:</span><span class="tt-val">${d.height?.toFixed(2)} m</span></div>
    <div class="tt-row"><span class="tt-label">Skóre:</span><span class="tt-val">${(d.score * 100)?.toFixed(1)}%</span></div>
    <div style="margin-top:4px;font-size:11px;color:#aaa">Hrany:<br>${edges}</div>
  `;
  tooltip.style.display = "block";
  tooltip.style.left = (event.clientX + 16) + "px";
  tooltip.style.top = (event.clientY - 10) + "px";
}

function hideTooltip() {
  tooltip.style.display = "none";
}

// ── Event handlers ────────────────────────────────────────────
let clickTimer = null;

renderer.domElement.addEventListener("pointermove", (event) => {
  if (draggingVertex) {
    // Drag vertex in XZ plane
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
      const sphere = draggingVertex;
      sphere.position.copy(intersection);
      // Update the vertex in editPlane
      const vi = sphere.userData.vertexIndex;
      editPlane.vertices3D[vi].copy(intersection);
      updateVertexSpheres(editPlane.vertices3D);
    }
    return;
  }

  // Hover
  const intersects = getIntersections(event);
  if (intersects.length > 0) {
    const po = getPlaneByMesh(intersects[0].object);
    if (po && po !== hoveredPlane) {
      if (hoveredPlane && hoveredPlane !== selectedPlane) {
        resetAllHighlight();
        if (selectedPlane) highlightPlane(selectedPlane, 0x44aaff);
      }
      hoveredPlane = po;
      if (po !== selectedPlane) highlightPlane(po, 0xaaccff);
    }
    if (hoveredPlane && hoveredPlane !== selectedPlane) {
      showTooltip(hoveredPlane, event);
    }
  } else {
    if (hoveredPlane && hoveredPlane !== selectedPlane) {
      resetAllHighlight();
      if (selectedPlane) highlightPlane(selectedPlane, 0x44aaff);
    }
    hoveredPlane = null;
    hideTooltip();
  }
});

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return; // left click only

  // Check vertex hit first
  if (editMode && editPlane) {
    const vertHits = getVertexIntersection(event);
    if (vertHits.length > 0) {
      draggingVertex = vertHits[0].object;
      controls.enabled = false;
      return;
    }
  }

  // Start click/dblclick detection
  const intersects = getIntersections(event);
  if (intersects.length > 0) {
    const po = getPlaneByMesh(intersects[0].object);
    if (!po) return;

    if (clickTimer) {
      // Double-click → toggle edit mode
      clearTimeout(clickTimer);
      clickTimer = null;
      handleDoubleClick(po);
    } else {
      clickTimer = setTimeout(() => {
        clickTimer = null;
        handleSingleClick(po, event);
      }, 280);
    }
  }
});

renderer.domElement.addEventListener("pointerup", () => {
  if (draggingVertex) {
    // Finalize vertex drag — rebuild plane
    rebuildPlane(editPlane);
    draggingVertex = null;
    controls.enabled = true;
  }
});

window.addEventListener("keydown", (event) => {
  if (!editMode || !editPlane) return;
  if (event.key === "Delete" || event.key === "Backspace") {
    // Delete last selected/hovered vertex
    if (draggingVertex) {
      const vi = draggingVertex.userData.vertexIndex;
      if (editPlane.vertices3D.length <= 3) {
        alert("Polygón musí mať aspoň 3 vrcholy.");
        return;
      }
      editPlane.vertices3D.splice(vi, 1);
      draggingVertex = null;
      controls.enabled = true;
      rebuildPlane(editPlane);
      createVertexSpheres(editPlane.vertices3D);
    }
  }
  if (event.key === "Escape") {
    exitEditMode();
  }
});

function handleSingleClick(po, event) {
  // Select plane
  resetAllHighlight();
  selectedPlane = po;
  highlightPlane(po, 0x44aaff);
  showTooltip(po, event);
}

function handleDoubleClick(po) {
  if (editMode && editPlane === po) {
    exitEditMode();
    return;
  }
  if (editMode) exitEditMode();
  enterEditMode(po);
}

function enterEditMode(po) {
  editMode = true;
  editPlane = po;
  editIndicator.classList.add("active");
  resetAllHighlight();
  highlightPlane(po, 0xff6600);
  // Deselect in orbit controls to avoid camera rotate on click+drag over vertices
  createVertexSpheres(po.vertices3D);
}

function exitEditMode() {
  editMode = false;
  editPlane = null;
  draggingVertex = null;
  editIndicator.classList.remove("active");
  removeVertexSpheres();
  resetAllHighlight();
  selectedPlane = null;
  controls.enabled = true;
}

// ── Resize ────────────────────────────────────────────────────
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// ── Render loop ───────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

console.log("RoofAIStudio 3D Viewer ready —", planes.length, "planes loaded.");
</script>
</body>
</html>'''
