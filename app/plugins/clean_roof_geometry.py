# Clean Roof Geometry v3 - fixed
import numpy as np, math
from scipy.spatial import ConvexHull
from sklearn.linear_model import RANSACRegressor

def simplify_polygon(pts2d, max_v=8):
    n = len(pts2d)
    if n <= max_v: return pts2d
    step = max(1, n // max_v)
    return pts2d[::step][:max_v]

class CleanRoofGeometry:
    def __init__(self, points, ground_z):
        self.pts0 = np.asarray(points)
        self.gz = float(ground_z)
        self.cx = self.pts0[:,0].mean(); self.cy = self.pts0[:,1].mean()
        self.pts = self.pts0.copy()
        self.pts[:,0] -= self.cx; self.pts[:,1] -= self.cy; self.pts[:,2] -= self.gz
        self.planes_out = []; self.edges_summary = {}
        self._extract()

    def _plane_basis(self, normal):
        z = np.array([0.,0.,1.]); u = np.cross(normal, z)
        un = np.linalg.norm(u)
        u = u/un if un > 1e-10 else np.array([1.,0.,0.])
        v = np.cross(normal, u); v /= np.linalg.norm(v)
        return u, v

    def _extract(self):
        remaining = np.ones(len(self.pts), dtype=bool); raw = []
        while remaining.sum() >= 20:
            p = self.pts[remaining]; X2 = p[:,:2]; Z = p[:,2]
            try:
                r = RANSACRegressor(residual_threshold=0.50, min_samples=max(8,int(len(p)*0.03)), max_trials=200, random_state=42)
                r.fit(X2, Z)
            except: break
            inl = r.inlier_mask_
            if inl.sum() < 20: break
            a,b = r.estimator_.coef_; c0 = r.estimator_.intercept_
            n = np.array([-a,-b,1.0]); n /= np.linalg.norm(n)
            pitch = math.degrees(math.acos(min(1, abs(n[2]))))
            idxs = np.where(remaining)[0][inl]; ipts = self.pts[idxs]
            u,v = self._plane_basis(n)
            p2d = np.column_stack([np.dot(ipts[:,:2], u[:2]), np.dot(ipts[:,:2], v[:2])])
            try:
                hull = ConvexHull(p2d); area2 = float(hull.volume); h2d = p2d[hull.vertices]
            except:
                area2 = (p2d[:,0].max()-p2d[:,0].min())*(p2d[:,1].max()-p2d[:,1].min())
                h2d = np.array([[p2d[:,0].min(),p2d[:,1].min()],[p2d[:,0].max(),p2d[:,1].min()],[p2d[:,0].max(),p2d[:,1].max()],[p2d[:,0].min(),p2d[:,1].max()]])
            area3 = area2 / max(abs(n[2]), 0.01)
            if area2 >= 2 and pitch >= 5:
                raw.append({'n':n,'a':a,'b':b,'c':c0,'pitch':round(float(pitch),1),'area3':round(float(area3),2),'area2':round(float(area2),2),'z_mean':float(ipts[:,2].mean()),'z_max':float(ipts[:,2].max()),'z_min':float(ipts[:,2].min()),'u':u,'v':v,'hull2d':h2d,'pts':len(ipts),'kind':'roof'})
            remaining[idxs] = False

        if remaining.sum() >= 20:
            low = self.pts[remaining]; low = low[low[:,2] < np.percentile(self.pts[:,2], 25)]
            if len(low) >= 20:
                try:
                    hull = ConvexHull(low[:,:2]); area2 = float(hull.volume); h2d = low[:,:2][hull.vertices]
                    if area2 >= 5:
                        raw.append({'n':np.array([0.,0.,1.]),'a':0,'b':0,'c':float(low[:,2].mean()),'pitch':0.,'area3':round(float(area2),2),'area2':round(float(area2),2),'z_mean':float(low[:,2].mean()),'z_max':float(low[:,2].max()),'z_min':float(low[:,2].min()),'u':np.array([1.,0.,0.]),'v':np.array([0.,1.,0.]),'hull2d':h2d,'pts':len(low),'kind':'platform'})
                except: pass

        self._build(raw)

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
            planes.append({'id':pid,'type':'plosina' if rp['kind']=='platform' else ('sedlova' if 15<=rp['pitch']<=45 else 'valbova'),'area_m2':rp['area3'],'pitch_deg':rp['pitch'],'z_min_m':rp['z_min'],'z_max_m':rp['z_max'],'vertices_3d':verts3d,'edges':edges})

        tn = {'o':'odkvap','n':'narozie','u':'uzlabie','h':'hreben','f':'stit','p':'plosina'}
        summary = {}
        for etype, lengths in esums.items():
            summary[etype] = {'nazov':tn.get(etype,etype),'pocet':len(lengths),'celkom_m':round(sum(lengths),3),'hrany':['{}{}'.format(etype,i+1) for i in range(len(lengths))]}
        self.planes_out = planes; self.edges_summary = summary

    def to_json(self, address=None):
        return {'address':address,'ground_z_m':self.gz,'planes':self.planes_out,'edges_summary':self.edges_summary}
