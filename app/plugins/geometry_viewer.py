#!/usr/bin/env python3
"""Generate 3D viewer HTML from RoofGeometry JSON — clean polygons with edge annotations."""
import json, os, base64

COLORS = {
    'o': 0xff4444,  # odkvap - red
    'h': 0x4488ff,  # hreben - blue
    'n': 0x44ff44,  # narozie - green
    'u': 0xffaa00,  # uzlabie - orange
    'f': 0x888888,  # stit - gray
    'p': 0xaa44ff,  # plosina - purple
}

def generate_geometry_viewer(geometry_json_path, orthophoto_path, output_path):
    """Generate self-contained HTML viewer showing clean roof polygon planes with edge callouts."""
    with open(geometry_json_path, 'r', encoding='utf-8') as f:
        geo = json.load(f)
    
    # Extract planes geometry
    planes_data = []
    for plane in geo.get('planes', []):
        verts = plane.get('vertices_3d', [])
        edges_out = []
        for e in plane.get('edges', []):
            v1_idx = e.get('v1', 0)
            v2_idx = e.get('v2', 0)
            if v1_idx < len(verts) and v2_idx < len(verts):
                edges_out.append({
                    'id': e.get('id', '?'),
                    'type': e.get('type', 'f'),
                    'length_m': e.get('length_m', 0),
                    'start': verts[v1_idx],
                    'end': verts[v2_idx],
                })
        if verts and len(verts) >= 3:
            planes_data.append({
                'id': plane.get('id', '?'),
                'type': plane.get('type', '?'),
                'area_m2': plane.get('area_m2', 0),
                'pitch_deg': plane.get('pitch_deg', 0),
                'vertices': verts,
                'edges': edges_out,
            })
    
    # Orthophoto as base64
    ortho_b64 = ''
    if orthophoto_path and os.path.exists(orthophoto_path):
        with open(orthophoto_path, 'rb') as f:
            ortho_b64 = base64.b64encode(f.read()).decode()
    
    addr = geo.get('address', 'Unknown')
    edges_summary = geo.get('edges_summary', {})
    names = {'o': 'Odkvap', 'n': 'Narozie', 'u': 'Uzlabie', 'h': 'Hreben', 'f': 'Stit', 'p': 'Plosina'}
    
    html = '''<!DOCTYPE html>
<html lang="sk">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Strecha - ''' + addr + '''</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;font-family:system-ui,sans-serif;overflow:hidden}
#container{width:100vw;height:100vh}
#panel{position:absolute;top:15px;right:15px;background:rgba(0,0,0,0.85);color:#ddd;padding:14px 18px;border-radius:10px;font-size:12px;min-width:200px;backdrop-filter:blur(10px);border:1px solid rgba(255,255,255,0.1);z-index:10;max-height:80vh;overflow-y:auto}
#panel h2{font-size:14px;color:#4fc3f7;margin-bottom:8px}
#panel .dim{display:flex;justify-content:space-between;padding:2px 0}
#panel .lbl{color:#999}#panel .val{color:#fff;font-weight:600}
#panel hr{border-color:rgba(255,255,255,0.08);margin:6px 0}
.etype{margin:4px 0 2px;font-weight:700;font-size:11px}
.edges{display:flex;flex-wrap:wrap;gap:3px;margin-bottom:6px}
.edge{font-size:10px;padding:2px 6px;border-radius:3px;background:rgba(255,255,255,0.08);cursor:pointer;transition:all 0.2s}
.edge:hover{background:rgba(255,255,255,0.2);transform:scale(1.05)}
.edge-o{border-left:2px solid #ff4444}.edge-h{border-left:2px solid #4488ff}
.edge-n{border-left:2px solid #44ff44}.edge-u{border-left:2px solid #ffaa00}
.edge-f{border-left:2px solid #888888}.edge-p{border-left:2px solid #aa44ff}
#hint{position:absolute;bottom:10px;left:50%;transform:translateX(-50%);color:rgba(255,255,255,0.3);font-size:11px}
#ortho{position:absolute;bottom:15px;left:15px;width:180px;border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,0.2);z-index:10}
#ortho img{width:100%;display:block}
#legend{position:absolute;bottom:15px;right:15px;font-size:10px;color:rgba(255,255,255,0.4);text-align:right;z-index:10}
</style></head>
<body>
<div id="container"></div>
<div id="panel">
<h2>''' + addr + '''</h2>
'''

    # Add edge summary
    for ek in ['o','h','n','u','f','p']:
        if ek in edges_summary:
            e = edges_summary[ek]
            html += '<div class="dim"><span class="lbl">' + names.get(ek,ek) + '</span><span class="val">{:.1f} m</span></div>'.format(e['celkom_m'])
    
    html += '''</div>
<div id="hint">LMB=otocenie | Pravy=posun | Koliesko=zoom | Klikni na hranu</div>
'''
    if ortho_b64:
        html += '<div id="ortho"><img src="data:image/jpeg;base64,' + ortho_b64[:8000] + '" /></div>'

    html += '<div id="legend">'
    for ek in ['o','h','n','u','f']:
        html += '<span style="color:#{:06x}">&#x25A0;</span> {} &nbsp; '.format(COLORS[ek], names.get(ek,ek))
    html += '</div>'

    html += '''
<script type="importmap">
{"imports":{"three":"./three.module.js","three/addons/":"./"}}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/CSS2DRenderer.js";

const container = document.getElementById("container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);

const camera = new THREE.PerspectiveCamera(50, container.clientWidth/container.clientHeight, 0.5, 200);
camera.position.set(30, 25, 20);
camera.lookAt(0, 0, 3);

const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
container.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(container.clientWidth, container.clientHeight);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
container.appendChild(labelRenderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 3);
controls.enableDamping = true;
controls.update();

// Lighting
scene.add(new THREE.AmbientLight(0x404070, 2));
const sun = new THREE.DirectionalLight(0xffffff, 3);
sun.position.set(30, 40, 20);
sun.castShadow = true;
sun.shadow.mapSize.set(1024, 1024);
scene.add(sun);

// Grid
const grid = new THREE.GridHelper(60, 30, 0x333355, 0x1a1a3e);
scene.add(grid);

// PLANES DATA
const planes = ''' + json.dumps(planes_data) + ''';

// Center all vertices
let allX = [], allY = [], allZ = [];
planes.forEach(p => p.vertices.forEach(v => { allX.push(v[0]); allY.push(v[1]); allZ.push(v[2]); }));
const cx = (Math.min(...allX) + Math.max(...allX)) / 2;
const cy = (Math.min(...allY) + Math.max(...allY)) / 2;
const cz = Math.min(...allZ);

// Render each plane
planes.forEach((plane, pi) => {
    const verts = plane.vertices;
    if (verts.length < 3) return;

    // Triangulate polygon (fan triangulation)
    const positions = [];
    for (let i = 1; i < verts.length - 1; i++) {
        positions.push(verts[0][0] - cx, verts[0][1] - cy, verts[0][2] - cz);
        positions.push(verts[i][0] - cx, verts[i][1] - cy, verts[i][2] - cz);
        positions.push(verts[i+1][0] - cx, verts[i+1][1] - cy, verts[i+1][2] - cz);
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geo.computeVertexNormals();

    // Semi-transparent face
    const hue = (pi * 50 + 30) % 360;
    const color = new THREE.Color().setHSL(hue/360, 0.5, 0.45);
    const mat = new THREE.MeshStandardMaterial({color, roughness:0.7, metalness:0.1, side:THREE.DoubleSide, transparent:true, opacity:0.75});
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    scene.add(mesh);

    // Wireframe outline
    const wireGeo = new THREE.BufferGeometry();
    const wireVerts = [];
    for (let i = 0; i < verts.length; i++) {
        const v1 = verts[i];
        const v2 = verts[(i+1) % verts.length];
        wireVerts.push(v1[0]-cx, v1[1]-cy, v1[2]-cz, v2[0]-cx, v2[1]-cy, v2[2]-cz);
    }
    wireGeo.setAttribute("position", new THREE.Float32BufferAttribute(wireVerts, 3));
    const wireMat = new THREE.LineBasicMaterial({color:0x000000, linewidth:1, transparent:true, opacity:0.3});
    scene.add(new THREE.LineSegments(wireGeo, wireMat));

    // Area label at centroid
    let cpx = 0, cpy = 0, cpz = 0;
    verts.forEach(v => { cpx += v[0]-cx; cpy += v[1]-cy; cpz += v[2]-cz; });
    cpx /= verts.length; cpy /= verts.length; cpz /= verts.length;
    
    const div = document.createElement("div");
    div.textContent = plane.id + "\\n" + plane.area_m2.toFixed(1) + "m2";
    div.style.cssText = "color:#fff;font-size:10px;background:rgba(0,0,0,0.7);padding:2px 6px;border-radius:3px;pointer-events:auto;cursor:default";
    const label = new CSS2DObject(div);
    label.position.set(cpx, cpy, cpz);
    scene.add(label);

    // Edge lines with colors
    plane.edges.forEach(edge => {
        const ec = ''' + json.dumps({k: v for k, v in COLORS.items()}) + '''[edge.type] || 0x888888;
        const geo2 = new THREE.BufferGeometry();
        geo2.setAttribute("position", new THREE.Float32BufferAttribute([
            edge.start[0]-cx, edge.start[1]-cy, edge.start[2]-cz,
            edge.end[0]-cx, edge.end[1]-cy, edge.end[2]-cz
        ], 3));
        const line = new THREE.Line(geo2, new THREE.LineBasicMaterial({color:ec, linewidth:2, transparent:true, opacity:0.9}));
        scene.add(line);

        // Small sphere at midpoint
        const mx = (edge.start[0] + edge.end[0])/2 - cx;
        const my = (edge.start[1] + edge.end[1])/2 - cy;
        const mz = (edge.start[2] + edge.end[2])/2 - cz;
        
        // Edge label
        const ediv = document.createElement("div");
        ediv.textContent = edge.id + " " + edge.length_m.toFixed(2) + "m";
        ediv.style.cssText = "color:#ccc;font-size:9px;background:rgba(0,0,0,0.8);padding:1px 4px;border-radius:2px;pointer-events:auto;cursor:default;border-left:2px solid #" + ec.toString(16).padStart(6,"0");
        const elabel = new CSS2DObject(ediv);
        elabel.position.set(mx, my, mz);
        scene.add(elabel);
    });
});

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
    labelRenderer.render(scene, camera);
}
animate();
window.addEventListener("resize", () => {
    camera.aspect = container.clientWidth/container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
    labelRenderer.setSize(container.clientWidth, container.clientHeight);
});
</script>
</body></html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return output_path
