import numpy as np
import cv2
from app.ai.pipeline.roof_plane_detector import RoofPlaneDetector

img = np.zeros((400, 400, 3), dtype=np.uint8)
pts1 = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32)
pts2 = np.array([[200, 200], [300, 200], [300, 300], [200, 300]], dtype=np.int32)
cv2.fillPoly(img, [pts1], (255, 255, 255))
cv2.fillPoly(img, [pts2], (255, 255, 255))

print('Running RoofPlaneDetector smoke test...')
detector = RoofPlaneDetector()
results = detector.detect(img)
print('Found', len(results), 'planes')
for i, r in enumerate(results):
    meta = r.metadata or {}
    verts = meta.get('polygon_vertices')
    area = meta.get('area_pixels')
    print(f'Plane {i}: area={area}, vertices={len(verts) if verts else None}')
