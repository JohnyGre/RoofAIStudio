#!/usr/bin/env python3
"""Roof geometry v6 — normal-based clustering + spatial sub-clusters + alpha shape."""
import numpy as np, math, json
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay

def alpha_shape(points_2d, alpha=None):
    """Concave hull from 2D points."""
    if len(points_2d) < 4:
        return convex_hull(points_2d)
    try:
        import alphashape
        if alpha is None:
            tree = KDTree(points_2d)
            dists, _ = tree.query(points_2d, k=5)
            alpha = float(np.percentile(dists[:, 1:].mean(axis=1), 90)) * 2
        shape = alphashape.alphashape(points_2d, alpha)
        if shape.geom_type == 'Polygon':
            return np.array(shape.exterior.coords)[:-1]
        elif shape.geom_type == 'MultiPolygon':
            largest = max(shape.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)[:-1]
    except: pass
    return convex_hull(points_2d)

def convex_hull(points_2d):
    try: return points_2d[ConvexHull(points_2d).vertices]
    except: return points_2d

def simplify_polygon(pts, max_v=10):
    n=len(pts)
    if n<=max_v: return pts
    step=max(1,n//max_v)
    return pts[::step][:max_v]

def classify_edge(v1,v2,pitch,zmin,zmax):
    dz=abs(v2[2]-v1[2]);mz=(v1[2]+v2[2])/2
    if pitch<3: return 'p'
    if dz<0.3 and mz<=zmin+0.5: return 'o'
    if dz<0.3 and mz>=zmax-0.5: return 'h'
    if dz<0.3: return 'o'
    return 'f'

def process_roof(points, ground_z):
    pts=np.asarray(points).copy()
    cx,cy=pts[:,0].mean(),pts[:,1].mean()
    pts[:,0]-=cx;pts[:,1]-=cy;pts[:,2]-=ground_z
    roof=pts[pts[:,2]>0.5]
    if len(roof)<50: return {'planes':[],'edges_summary':{}}
    
    # Downsample for normal computation
    n=min(len(roof),5000)
    idx=np.random.choice(len(roof),n,replace=False)
    sample=roof[idx]
    
    # 1. PCA normals per point
    tree=KDTree(sample[:,:2])
    normals=np.zeros((len(sample),3))
    for i in range(len(sample)):
        nb=tree.query_radius(sample[i:i+1,:2],r=1.5)[0]
        if len(nb)>=8:
            pca=PCA(n_components=3).fit(sample[nb])
            n=pca.components_[2]
            if n[2]<0:n=-n
            normals[i]=n
        else:normals[i]=[0,0,1]
    
    # 2. Cluster by normal direction
    feats=StandardScaler().fit_transform(np.column_stack([normals,sample[:,2:]/5.0]))
    norm_labels=DBSCAN(eps=0.3,min_samples=20).fit_predict(feats)
    
    planes_out=[]
    etypes={'o':0,'n':0,'u':0,'h':0,'f':0,'p':0}
    esums={}
    
    for cid in sorted(set(norm_labels)):
        if cid==-1:continue
        cmask=norm_labels==cid
        pts_n=sample[cmask]
        avg_n=normals[cmask].mean(axis=0)
        avg_n/=np.linalg.norm(avg_n)
        pitch=math.degrees(math.acos(min(1,abs(avg_n[2]))))
        if pitch<5:continue
        
        # 3. Spatial sub-clustering within this normal group
        # Use neighborhood from FULL roof points near these points
        cm=np.median(pts_n,axis=0)
        radius=max(pts_n[:,0].max()-pts_n[:,0].min(),pts_n[:,1].max()-pts_n[:,1].min())/2+3
        radius=max(radius,5)
        nearby=roof[np.sqrt((roof[:,0]-cm[0])**2+(roof[:,1]-cm[1])**2)<radius]
        
        # Spatial DBSCAN on nearby points
        xy_scaled=StandardScaler().fit_transform(nearby[:,:2])
        try:
            sp_labels=DBSCAN(eps=0.15,min_samples=15).fit_predict(xy_scaled)
        except:continue
        
        for did in sorted(set(sp_labels)):
            if did==-1:continue
            dmask=sp_labels==did;dpts=nearby[dmask]
            if len(dpts)<20:continue
            
            # Fit plane
            try:
                coeffs,_,_,_=np.linalg.lstsq(np.column_stack([dpts[:,:2],np.ones(len(dpts))]),dpts[:,2],rcond=None)
                a,b,c0=coeffs;pl_n=np.array([-a,-b,1.0]);pl_n/=np.linalg.norm(pl_n)
                pl_pitch=math.degrees(math.acos(min(1,abs(pl_n[2]))))
            except:continue
            
            if pl_pitch<5:continue
            
            # Build 2D basis
            z_ax=np.array([0.,0.,1.])
            u=np.cross(pl_n,z_ax);un=np.linalg.norm(u)
            u=u/un if un>1e-10 else np.array([1.,0.,0.])
            v=np.cross(pl_n,u);v/=np.linalg.norm(v)
            p2d=np.column_stack([np.dot(dpts[:,:2],u[:2]),np.dot(dpts[:,:2],v[:2])])
            
            # Alpha shape
            bnd=alpha_shape(p2d)
            if len(bnd)<4:bnd=convex_hull(p2d)
            s2d=simplify_polygon(bnd,max_v=8)
            if len(s2d)<3:continue
            
            # 3D vertices
            verts=[]
            for pt2 in s2d:
                xy=pt2[0]*u[:2]+pt2[1]*v[:2]
                z=float(dpts[:,2].mean())
                verts.append([round(float(xy[0]),3),round(float(xy[1]),3),round(z,3)])
            zs=[v[2] for v in verts];zmin,zmax=min(zs),max(zs)
            
            # Area
            try:
                from shapely.geometry import Polygon
                a2d=Polygon(s2d).area
            except:a2d=float(ConvexHull(s2d).volume)
            a3d=a2d/max(abs(pl_n[2]),0.01)
            if a3d<3:continue
            
            # Edges
            edges=[]
            for ei in range(len(verts)):
                v1=np.array(verts[ei]);v2=np.array(verts[(ei+1)%len(verts)])
                elen=float(np.linalg.norm(v2-v1))
                if elen<0.3:continue
                etype=classify_edge(v1,v2,pl_pitch,zmin,zmax)
                nid=etypes[etype];etypes[etype]+=1
                edges.append({'id':f'{etype}{nid+1}','type':etype,'length_m':round(elen,3),'v1':ei,'v2':(ei+1)%len(verts)})
                esums.setdefault(etype,[]).append(round(elen,3))
            
            planes_out.append({
                'id':f'R{len(planes_out)+1}','type':'sedlova' if 15<=pl_pitch<=45 else 'valbova',
                'area_m2':round(float(a3d),2),'pitch_deg':round(float(pl_pitch),1),
                'z_min_m':round(float(dpts[:,2].min()),2),'z_max_m':round(float(dpts[:,2].max()),2),
                'vertices_3d':verts,'edges':edges})
    
    # Overlap filter
    if planes_out:
        planes_out.sort(key=lambda p:p['area_m2'],reverse=True)
        filt=[]
        for p in planes_out:
            dup=False
            for k in filt:
                vp=np.array(p['vertices_3d'])
                vk=np.array(k['vertices_3d'])
                if vp.shape==vk.shape:
                    diff=np.abs(vp-vk).mean()
                    if diff<2:dup=True;break
            if not dup:filt.append(p)
        planes_out=filt[:12]
    
    # Platform
    low=pts[pts[:,2]<1.5]
    if len(low)>20:
        try:
            h=ConvexHull(low[:,:2]);a2d=float(h.volume)
            if a2d>=3:
                verts=[[round(float(x),3),round(float(y),3),round(float(low[:,2].mean()),3)] for x,y in low[:,:2][h.vertices]]
                edges=[]
                for ei in range(len(verts)):
                    v1=np.array(verts[ei]);v2=np.array(verts[(ei+1)%len(verts)])
                    elen=float(np.linalg.norm(v2-v1))
                    if elen>=0.3:
                        nid=etypes['p'];etypes['p']+=1
                        edges.append({'id':f'p{nid+1}','type':'p','length_m':round(elen,3),'v1':ei,'v2':(ei+1)%len(verts)})
                        esums.setdefault('p',[]).append(round(elen,3))
                planes_out.append({'id':'P1','type':'plosina','area_m2':round(a2d,2),'pitch_deg':0.0,
                    'z_min_m':round(float(low[:,2].min()),2),'z_max_m':round(float(low[:,2].max()),2),
                    'vertices_3d':verts,'edges':edges})
        except:pass
    
    tn={'o':'odkvap','n':'narozie','u':'uzlabie','h':'hreben','f':'stit','p':'plosina'}
    summary={et:{'nazov':tn.get(et,et),'pocet':len(lengths),'celkom_m':round(sum(lengths),3),
        'hrany':[f'{et}{i+1}' for i in range(len(lengths))]} for et,lengths in esums.items()}
    
    return {'planes':planes_out,'edges_summary':summary}

class CleanRoofGeometry:
    def __init__(self,points,ground_z):
        self.result=process_roof(np.asarray(points),float(ground_z))
        self.planes_out=self.result['planes']
        self.edges_summary=self.result['edges_summary']
        self.gz=float(ground_z)
    def to_json(self,address=None):
        return {'address':address,'ground_z_m':self.gz,'planes':self.planes_out,'edges_summary':self.edges_summary}
