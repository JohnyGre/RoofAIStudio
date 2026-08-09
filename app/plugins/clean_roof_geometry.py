#!/usr/bin/env python3
"""Clean Roof Geometry v4 — uses PCA normals instead of RANSAC for better pitch detection."""
import numpy as np, math
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def simplify_polygon(pts2d, max_v=8):
    n = len(pts2d)
    if n <= max_v: return pts2d
    step = max(1, n // max_v)
    return pts2d[::step][:max_v]

class CleanRoofGeometry:
    def __init__(self, points, ground_z):
        self.gz = float(ground_z)
        self.cx = points[:,0].mean(); self.cy = points[:,1].mean()
        self.pts = points.copy()
        self.pts[:,0] -= self.cx; self.pts[:,1] -= self.cy; self.pts[:,2] -= self.gz
        self.planes_out = []; self.edges_summary = {}
        self._extract()

    def _extract(self):
        # Downsample for speed
        n = min(len(self.pts), 8000)
        idx = np.random.choice(len(self.pts), n, replace=False)
        sample = self.pts[idx]
        
        # Compute PCA normals for each point
        tree = KDTree(sample[:, :2])
        normals = np.zeros((len(sample), 3))
        for i in range(len(sample)):
            nb = tree.query_radius(sample[i:i+1, :2], r=1.5)[0]
            if len(nb) >= 8:
                pca = PCA(n_components=3).fit(sample[nb])
                n = pca.components_[2]
                if n[2] < 0: n = -n
                normals[i] = n
            else:
                normals[i] = [0, 0, 1]
        
        # Cluster by normal direction + Z height
        features = np.column_stack([normals, sample[:, 2:] / 5.0])  # Scale Z
        features_s = StandardScaler().fit_transform(features)
        labels = DBSCAN(eps=0.35, min_samples=25).fit_predict(features_s)
        
        # For each normal cluster, fit a plane and get convex hull
        raw = []
        for cid in sorted(set(labels)):
            if cid == -1: continue
            mask = labels == cid
            cluster_pts = sample[mask]
            avg_normal = normals[mask].mean(axis=0)
            avg_normal /= np.linalg.norm(avg_normal)
            pitch = math.degrees(math.acos(min(1, abs(avg_normal[2]))))
            
            if pitch < 6:  # Skip near-flat (ground)
                continue
            
            # Get all points NEAR this cluster (from full point cloud)
            cm = cluster_pts.mean(axis=0)
            radius = max(cluster_pts[:, 0].ptp() if hasattr(cluster_pts[:, 0], 'ptp') else 
                        float(cluster_pts[:, 0].max() - cluster_pts[:, 0].min()),
                        cluster_pts[:, 1].max() - cluster_pts[:, 1].min()) / 2 + 2
            radius = max(radius, 5)
            
            nearby_mask = np.sqrt((self.pts[:, 0] - cm[0])**2 + (self.pts[:, 1] - cm[1])**2) < radius
            region_pts = self.pts[nearby_mask]
            
            # Fit plane directly from cluster points
            # Project to 2D in plane's basis
            z_ax = np.array([0.,0.,1.])
            u = np.cross(avg_normal, z_ax)
            un = np.linalg.norm(u)
            u = u/un if un > 1e-10 else np.array([1.,0.,0.])
            v = np.cross(avg_normal, u); v /= np.linalg.norm(v)
            
            # Also get nearby points from full cloud for denser sampling
            proj_all = np.column_stack([np.dot(self.pts[:,:2], u[:2]), np.dot(self.pts[:,:2], v[:2])])
            cproj = np.column_stack([np.dot(cluster_pts[:,:2], u[:2]), np.dot(cluster_pts[:,:2], v[:2])])
            ch_x0, ch_x1 = cproj[:,0].min() - 1, cproj[:,0].max() + 1
            ch_y0, ch_y1 = cproj[:,1].min() - 1, cproj[:,1].max() + 1
            cmask = (proj_all[:,0] >= ch_x0) & (proj_all[:,0] <= ch_x1) & (proj_all[:,1] >= ch_y0) & (proj_all[:,1] <= ch_y1)
            dpts = self.pts[cmask]
            if len(dpts) < 20:
                dpts = cluster_pts
            
            p2d = np.column_stack([np.dot(dpts[:,:2], u[:2]), np.dot(dpts[:,:2], v[:2])])
            
            try:
                hull = ConvexHull(p2d); area2 = float(hull.volume); h2d = p2d[hull.vertices]
            except:
                area2 = (p2d[:,0].max()-p2d[:,0].min())*(p2d[:,1].max()-p2d[:,1].min())
                if area2 < 2: continue
                h2d = np.array([[p2d[:,0].min(),p2d[:,1].min()],[p2d[:,0].max(),p2d[:,1].min()],
                                [p2d[:,0].max(),p2d[:,1].max()],[p2d[:,0].min(),p2d[:,1].max()]])
            
            if area2 < 3: continue
            
            # Fit plane equation
            A = np.column_stack([dpts[:,:2], np.ones(len(dpts))])
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, dpts[:,2], rcond=None)
                a, b, c_val = coeffs
                plane_n = np.array([-a, -b, 1.0]); plane_n /= np.linalg.norm(plane_n)
                plane_pitch = math.degrees(math.acos(min(1, abs(plane_n[2]))))
            except:
                plane_n = avg_normal; plane_pitch = pitch; c_val = 0
            
            area3 = area2 / max(abs(plane_n[2]), 0.01)
            
            raw.append({
                'n':plane_n, 'a':float(a), 'b':float(b), 'c':float(c_val),
                'pitch':round(float(plane_pitch),1),
                'area3':round(float(area3),2),'area2':round(float(area2),2),
                'z_mean':float(dpts[:,2].mean()),'z_max':float(dpts[:,2].max()),'z_min':float(dpts[:,2].min()),
                'u':u,'v':v,'hull2d':h2d,'pts':len(dpts),'kind':'roof'})
        
        # Platform detection (flat, low points not covered)
        try:
            all_used = np.zeros(len(self.pts), dtype=bool)
            for rp in raw:
                h = rp['hull2d']; u,v = rp['u'], rp['v']
                proj = np.column_stack([np.dot(self.pts[:,:2],u[:2]), np.dot(self.pts[:,:2],v[:2])])
                in_b = (proj[:,0]>=h[:,0].min()-1)&(proj[:,0]<=h[:,0].max()+1)&(proj[:,1]>=h[:,1].min()-1)&(proj[:,1]<=h[:,1].max()+1)
                all_used |= in_b
            low = self.pts[~all_used]
            low = low[low[:,2] < np.percentile(self.pts[:,2], 20)]
            if len(low) >= 20:
                hull = ConvexHull(low[:,:2]); area2 = float(hull.volume)
                if area2 >= 5:
                    raw.append({'n':np.array([0.,0.,1.]),'a':0,'b':0,'c':float(low[:,2].mean()),
                        'pitch':0.,'area3':round(float(area2),2),'area2':round(float(area2),2),
                        'z_mean':float(low[:,2].mean()),'z_max':float(low[:,2].max()),'z_min':float(low[:,2].min()),
                        'u':np.array([1.,0.,0.]),'v':np.array([0.,1.,0.]),'hull2d':low[:,:2][hull.vertices],
                        'pts':len(low),'kind':'platform'})
        except: pass
        
        # Filter overlapping planes with similar normals
        if raw:
            raw.sort(key=lambda x: x['area2'], reverse=True)
            filtered = []
            for rp in raw:
                dup = False
                for kept in filtered:
                    angle = math.degrees(math.acos(min(1, abs(np.dot(kept['n'], rp['n'])))))
                    if angle < 15:
                        # Check XY overlap
                        overlap = self._overlap_ratio(kept['hull2d'], rp['hull2d'])
                        if overlap > 0.5:
                            dup = True; break
                if not dup:
                    filtered.append(rp)
            raw = filtered[:14]
        
        self._build(raw)
    
    def _overlap_ratio(self, hull_a, hull_b):
        try:
            hxa0, hxa1 = hull_a[:,0].min(), hull_a[:,0].max()
            hya0, hya1 = hull_a[:,1].min(), hull_a[:,1].max()
            hxb0, hxb1 = hull_b[:,0].min(), hull_b[:,0].max()
            hyb0, hyb1 = hull_b[:,1].min(), hull_b[:,1].max()
            dx = max(0, min(hxa1,hxb1) - max(hxa0,hxb0))
            dy = max(0, min(hya1,hyb1) - max(hya0,hyb0))
            b_area = (hxb1-hxb0)*(hyb1-hyb0)
            return (dx*dy)/b_area if b_area > 0 else 0
        except: return 0

    def _build(self, raw):
        planes = []; etypes = {'o':0,'n':0,'u':0,'h':0,'f':0,'p':0}; esums = {}
        for pi, rp in enumerate(raw):
            s2d = simplify_polygon(rp['hull2d'], max_v=8)
            u,v,n0 = rp['u'], rp['v'], rp['n']
            verts3d = []
            for pt2 in s2d:
                xy = pt2[0]*u[:2] + pt2[1]*v[:2]
                if abs(n0[2]) > 0.001:
                    z = (rp['c']*n0[2] - n0[0]*xy[0] - n0[1]*xy[1]) / n0[2]
                else: z = rp['z_mean']
                verts3d.append([round(float(xy[0]),3),round(float(xy[1]),3),round(float(z),3)])
            if not verts3d: continue
            zs = [v[2] for v in verts3d]; zmin,zmax = min(zs), max(zs)
            edges = []
            for ei in range(len(verts3d)):
                v1 = np.array(verts3d[ei]); v2 = np.array(verts3d[(ei+1)%len(verts3d)])
                elen = float(np.linalg.norm(v2-v1)); dz = abs(v2[2]-v1[2]); mz = (v1[2]+v2[2])/2
                if rp['kind'] == 'platform': etype = 'p'
                elif dz < 0.3 and mz <= zmin + 0.5: etype = 'o'
                elif dz < 0.3 and mz >= zmax - 0.5: etype = 'h'
                elif dz < 0.3: etype = 'o'
                else: etype = 'f'
                nid = etypes[etype]; etypes[etype] += 1
                edges.append({'id':'{}{}'.format(etype,nid+1),'type':etype,'length_m':round(elen,3),'v1':ei,'v2':(ei+1)%len(verts3d)})
                esums.setdefault(etype, []).append(round(elen,3))
            pid = 'P{}'.format(pi+1) if rp['kind']=='platform' else 'R{}'.format(pi+1)
            planes.append({
                'id':pid,'type':'plosina' if rp['kind']=='platform' else ('sedlova' if 15<=rp['pitch']<=45 else 'valbova'),
                'area_m2':rp['area3'],'pitch_deg':rp['pitch'],'z_min_m':rp['z_min'],'z_max_m':rp['z_max'],
                'vertices_3d':verts3d,'edges':edges})
        tn = {'o':'odkvap','n':'narozie','u':'uzlabie','h':'hreben','f':'stit','p':'plosina'}
        summary = {}
        for etype, lengths in esums.items():
            summary[etype] = {'nazov':tn.get(etype,etype),'pocet':len(lengths),'celkom_m':round(sum(lengths),3),
                'hrany':['{}{}'.format(etype,i+1) for i in range(len(lengths))]}
        self.planes_out = planes; self.edges_summary = summary

    def to_json(self, address=None):
        return {'address':address,'ground_z_m':self.gz,'planes':self.planes_out,'edges_summary':self.edges_summary}
