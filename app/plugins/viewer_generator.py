#!/usr/bin/env python3
"""Generate a 3D viewer HTML with smoothed model + dimension annotations."""
import json, os, math
import numpy as np

def generate_viewer(ply_path, obj_path, orthophoto_path, meta, output_path):
    """
    Generate a self-contained HTML 3D viewer with:
    - Smoothed 3D model via three.js
    - Dimension annotations (length, width, height)
    - Orbit controls, measurement overlay
    """
    # Read PLY vertices and faces
    vertices, faces = _read_ply(ply_path)
    if vertices is None:
        return None
    
    # Smooth mesh (simple Laplacian)
    vertices_smooth = _laplacian_smooth(vertices, faces, iterations=3)
    
    # Write smoothed PLY
    smooth_ply = ply_path.replace('.ply', '_smooth.ply')
    _write_ply(smooth_ply, vertices_smooth, faces)
    
    # Write smoothed OBJ
    smooth_obj = obj_path.replace('.obj', '_smooth.obj')
    _write_obj(smooth_obj, vertices_smooth, faces)
    
    # Calculate bounding box for dimensions
    bbox = {
        'min_x': float(vertices_smooth[:, 0].min()),
        'max_x': float(vertices_smooth[:, 0].max()),
        'min_y': float(vertices_smooth[:, 1].min()),
        'max_y': float(vertices_smooth[:, 1].max()),
        'min_z': float(vertices_smooth[:, 2].min()),
        'max_z': float(vertices_smooth[:, 2].max()),
    }
    bbox['width'] = bbox['max_x'] - bbox['min_x']
    bbox['depth'] = bbox['max_y'] - bbox['min_y']
    bbox['height_min'] = bbox['min_z']
    bbox['height_max'] = bbox['max_z']
    
    # Center model
    cx = (bbox['min_x'] + bbox['max_x']) / 2
    cy = (bbox['min_y'] + bbox['max_y']) / 2
    cz = bbox['min_z']
    v_centered = vertices_smooth.copy()
    v_centered[:, 0] -= cx
    v_centered[:, 1] -= cy  
    v_centered[:, 2] -= cz
    
    # Write centered PLY for viewer
    viewer_ply = ply_path.replace('.ply', '_viewer.ply')
    _write_ply(viewer_ply, v_centered, faces)
    
    # Generate HTML
    html = _build_html(viewer_ply, orthophoto_path, meta, bbox, v_centered, faces)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return {
        'smooth_ply': smooth_ply,
        'smooth_obj': smooth_obj,
        'viewer_html': output_path,
        'dimensions': {
            'width_m': round(bbox['width'], 2),
            'depth_m': round(bbox['depth'], 2),
            'ridge_height_m': round(bbox['height_max'] - bbox['height_min'], 2),
            'eave_height_m': round(bbox['height_min'], 2),
            'area_m2': round(bbox['width'] * bbox['depth'], 1),
        }
    }


def _read_ply(path):
    """Read PLY ASCII file."""
    with open(path) as f:
        lines = f.readlines()
    
    # Parse header
    n_vertices = 0
    n_faces = 0
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            n_vertices = int(line.split()[-1])
        elif line.startswith('element face'):
            n_faces = int(line.split()[-1])
        elif line.startswith('end_header'):
            header_end = i
            break
    
    vertices = np.zeros((n_vertices, 3))
    for i in range(n_vertices):
        parts = lines[header_end + 1 + i].split()
        vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    
    faces = []
    for i in range(n_faces):
        parts = lines[header_end + 1 + n_vertices + i].split()
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    
    return vertices, faces


def _laplacian_smooth(vertices, faces, iterations=3):
    """Simple Laplacian mesh smoothing."""
    from collections import defaultdict
    
    v = vertices.copy()
    for _ in range(iterations):
        neighbors = defaultdict(list)
        for f in faces:
            for i in range(3):
                neighbors[f[i]].extend([f[(i+1)%3], f[(i+2)%3]])
        
        v_new = v.copy()
        for idx in range(len(v)):
            if idx in neighbors and neighbors[idx]:
                nbrs = list(set(neighbors[idx]))
                if len(nbrs) > 2:
                    avg = np.mean(v[nbrs], axis=0)
                    v_new[idx] = v[idx] * 0.5 + avg * 0.5
        v = v_new
    return v


def _write_ply(path, vertices, faces):
    with open(path, 'w') as f:
        f.write('ply\nformat ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(vertices)))
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('element face {}\n'.format(len(faces)))
        f.write('property list uchar int vertex_indices\nend_header\n')
        for v in vertices:
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
        for fc in faces:
            f.write('3 {} {} {}\n'.format(fc[0], fc[1], fc[2]))


def _write_obj(path, vertices, faces):
    with open(path, 'w') as f:
        f.write('# RoofAIStudio smoothed mesh\n')
        for v in vertices:
            f.write('v {:.4f} {:.4f} {:.4f}\n'.format(v[0], v[1], v[2]))
        for fc in faces:
            f.write('f {} {} {}\n'.format(fc[0]+1, fc[1]+1, fc[2]+1))


def _build_html(viewer_ply, orthophoto_path, meta, bbox, vertices, faces):
    """Build self-contained HTML viewer with three.js."""
    # Convert vertices + faces to JSON for embedding
    v_flat = vertices.flatten().tolist()
    f_flat = [idx for face in faces for idx in face]
    
    # Orthophoto as base64
    import base64
    ortho_b64 = ''
    if os.path.exists(orthophoto_path):
        with open(orthophoto_path, 'rb') as f:
            ortho_b64 = base64.b64encode(f.read()).decode()
    
    w = bbox['width']
    d = bbox['depth']
    h = bbox['height_max'] - bbox['height_min']
    ridge_h = round(meta.get('height_ridge', h), 2)
    eave_h = round(meta.get('height_eave', bbox['height_min']), 2)
    pitch = round(meta.get('pitch', 0), 1)
    addr = meta.get('address', 'Unknown')
    area = meta.get('area', round(w * d, 1))
    
    html = '''<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>3D Model - ''' + addr + '''</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#1a1a2e; font-family:system-ui,sans-serif; overflow:hidden; }
#container { width:100vw; height:100vh; }
#panel {
  position:absolute; top:20px; right:20px; background:rgba(0,0,0,0.85);
  color:#eee; padding:18px 22px; border-radius:12px; 
  font-size:13px; min-width:240px; backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,0.1); z-index:10;
}
#panel h2 { font-size:15px; margin-bottom:10px; color:#4fc3f7; }
#panel .dim { display:flex; justify-content:space-between; padding:3px 0; }
#panel .dim span:first-child { color:#999; }
#panel .dim span:last-child { color:#fff; font-weight:600; font-variant-numeric:tabular-nums; }
#panel hr { border-color:rgba(255,255,255,0.1); margin:8px 0; }
#hint { position:absolute; bottom:20px; left:50%; transform:translateX(-50%);
  color:rgba(255,255,255,0.4); font-size:12px; }
#ortho-overlay {
  position:absolute; bottom:20px; left:20px; width:220px; 
  border-radius:8px; overflow:hidden; border:1px solid rgba(255,255,255,0.2);
  z-index:10;
}
#ortho-overlay img { width:100%; display:block; }
</style>
</head>
<body>
<div id="container"></div>
<div id="panel">
  <h2>''' + addr + '''</h2>
  <div class="dim"><span>Pôdorys</span><span>''' + '{:.1f} x {:.1f} m'.format(w, d) + '''</span></div>
  <div class="dim"><span>Plocha</span><span>''' + '{:.0f} m²'.format(area) + '''</span></div>
  <hr>
  <div class="dim"><span>Výška hrebeňa</span><span>''' + '{:.2f} m'.format(ridge_h) + '''</span></div>
  <div class="dim"><span>Výška odkvapu</span><span>''' + '{:.2f} m'.format(eave_h) + '''</span></div>
  <hr>
  <div class="dim"><span>Sklon</span><span>''' + '{:.0f}°'.format(pitch) + '''</span></div>
  <div class="dim"><span>Typ</span><span>''' + meta.get('roof_type', '?') + '''</span></div>
</div>
''' + ('''
<div id="ortho-overlay"><img src="data:image/jpeg;base64,''' + ortho_b64[:10000] + '''" /></div>
''' if ortho_b64 else '') + '''
<div id="hint">🖱️ LMB=otáčanie | Pravý=posun | Koliesko=zoom</div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const container = document.getElementById("container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
scene.fog = new THREE.Fog(0x1a1a2e, 50, 200);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.5, 500);
camera.position.set(''' + '{:.1f},{:.1f},{:.1f}'.format(w*1.5, d*1.5, h*2+10) + ''');
camera.lookAt(0, 0, ''' + '{:.1f}'.format(h/2) + ''');

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, ''' + '{:.1f}'.format(h/2) + ''');
controls.enableDamping = true;
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0x404060, 2));
const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.position.set(''' + '{:.1f},{:.1f},{:.1f}'.format(w, d, h+20) + ''');
sun.castShadow = true;
sun.shadow.mapSize.set(2048,2048);
scene.add(sun);
scene.add(new THREE.DirectionalLight(0x6688cc, 1));

// Grid
const grid = new THREE.GridHelper(''' + '{:.0f}'.format(max(w,d)*2) + ''', 20, 0x444466, 0x222244);
scene.add(grid);

// Build mesh from vertices/faces
const v = ''' + json.dumps(v_flat) + ''';
const f = ''' + json.dumps(f_flat) + ''';
const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.Float32BufferAttribute(v, 3));
geo.setIndex(f);
geo.computeVertexNormals();

const mat = new THREE.MeshStandardMaterial({
  color: 0xe8734a, roughness: 0.6, metalness: 0.1,
  flatShading: false, side: THREE.DoubleSide
});
const mesh = new THREE.Mesh(geo, mat);
mesh.castShadow = true;
mesh.receiveShadow = true;
scene.add(mesh);

// Wireframe
const wireMat = new THREE.MeshBasicMaterial({color:0x000000, wireframe:true, opacity:0.08, transparent:true});
const wire = new THREE.Mesh(geo, wireMat);
scene.add(wire);

// Ground plane
const groundGeo = new THREE.PlaneGeometry(''' + '{:.0f}'.format(max(w,d)*3) + ''', ''' + '{:.0f}'.format(max(w,d)*3) + ''');
const groundMat = new THREE.MeshStandardMaterial({color:0x2d5a27, roughness:0.9});
const ground = new THREE.Mesh(groundGeo, groundMat);
ground.rotation.x = -Math.PI/2;
ground.position.y = -0.05;
ground.receiveShadow = true;
scene.add(ground);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
window.addEventListener("resize", () => {
  camera.aspect = container.clientWidth/container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});
</script>
</body>
</html>'''
    return html
