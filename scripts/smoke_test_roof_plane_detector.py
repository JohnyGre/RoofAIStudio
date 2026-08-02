import numpy as np
import cv2
import math
from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector

# Create an image with an L-shaped (concave) polygon and a rectangle
img = np.zeros((400, 400, 3), dtype=np.uint8)
# L-shape: points form a concave polygon
l_shape = np.array([[50,50], [150,50], [150,100], [100,100], [100,150], [50,150]], dtype=np.int32)
rect = np.array([[250,250], [350,250], [350,350], [250,350]], dtype=np.int32)
cv2.fillPoly(img, [l_shape], (255,255,255))
cv2.fillPoly(img, [rect], (255,255,255))

print('Running RoofPlaneDetector smoke test (includes L-shaped concave polygon)...')
detector = RoofPlaneDetector()
results = detector.detect(img)
print('Found', len(results), 'planes')

# Helper: detect if polygon has at least one reflex (concave) vertex
def has_concavity(verts):
    if not verts or len(verts) < 4:
        return False
    # compute signed angles via cross products
    sign_changes = 0
    signs = []
    n = len(verts)
    for i in range(n):
        p_prev = verts[(i-1) % n]
        p_curr = verts[i]
        p_next = verts[(i+1) % n]
        v1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
        v2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])
        cross = v1[0]*v2[1] - v1[1]*v2[0]
        signs.append(cross)
    # If any cross product has opposite sign to majority, polygon is non-convex
    positive = sum(1 for s in signs if s > 0)
    negative = sum(1 for s in signs if s < 0)
    return positive > 0 and negative > 0

for i, r in enumerate(results):
    meta = r.metadata or {}
    verts = meta.get('polygon_vertices')
    area = meta.get('area_pixels')
    concave = has_concavity(verts)
    print(f'Plane {i}: area={area}, vertices={len(verts) if verts else None}, concave_preserved={concave}')

# Check expected: at least one detected polygon should preserve concavity
if any(has_concavity(r.metadata.get('polygon_vertices')) for r in results):
    print('SUCCESS: Concavity preserved in at least one detected polygon.')
else:
    print('FAIL: No concave polygons preserved; convex hull may have been used.')
