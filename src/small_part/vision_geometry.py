"""RGB geometry only: existing cyan liquid surface and orange rack apertures.

No simulator state, segmentation IDs, depth images, reference trajectories, or
object poses are accepted. Dimensions and camera calibration are known priors.
The color/shape assumptions are specific to this simulated lab scene.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import label,find_objects,binary_fill_holes,binary_closing,binary_erosion,binary_dilation
from scipy.optimize import least_squares

@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    rotation: np.ndarray  # camera-local axes in world; looks along -Z
    fovy: float
    width: int
    height: int
    @property
    def focal(self):return self.height/(2*np.tan(np.deg2rad(self.fovy)/2))
    def project(self,points):
        c=(np.asarray(points)-self.position)@self.rotation
        z=-c[...,2]
        return np.stack([self.width/2+self.focal*c[...,0]/z,self.height/2-self.focal*c[...,1]/z],axis=-1)
    def ray(self,uv):
        x,y=np.asarray(uv);v=self.rotation@np.array([(x-self.width/2)/self.focal,-(y-self.height/2)/self.focal,-1.])
        return v/np.linalg.norm(v)
    def plane(self,uv,height):
        v=self.ray(uv)
        return self.position+v*((height-self.position[2])/v[2])

def components(mask,min_area=8):
    lab,n=label(mask);out=[]
    for i,sl in enumerate(find_objects(lab),1):
        yy,xx=np.where(lab[sl]==i);area=len(xx)
        if area<min_area:continue
        xy=np.c_[xx+sl[1].start+.5,yy+sl[0].start+.5]
        sub=lab[sl]==i;edge=sub&~binary_erosion(sub)
        ey,ex=np.where(edge)
        out.append(dict(area=area,center=xy.mean(0),pixels=xy,boundary=np.c_[ex+sl[1].start+.5,ey+sl[0].start+.5],width=sl[1].stop-sl[1].start,height=sl[0].stop-sl[0].start))
    return out

def liquid_surface(rgb):
    r,g,b=np.asarray(rgb,dtype=float).transpose(2,0,1)
    mask=(g>r*1.3)&(b>r*1.3)&(b<g*1.12)&(g>120)
    found=components(mask,8)
    found=[c for c in found if c['width']>=4 and c['height']>=3]
    return max(found,key=lambda c:c['area']) if found else None

def orange_holes(rgb,camera):
    r,g,b=np.asarray(rgb,dtype=float).transpose(2,0,1)
    mask=(r>g*1.22)&(g>b*1.35)&(r>100)&(g>50)
    mask=binary_closing(mask,iterations=1)
    found=components(binary_fill_holes(mask)&~mask,30)
    choices=[]
    for c in found:
        xy=c['pixels'];cov=np.cov(xy.T);ev=np.linalg.eigvalsh(cov)
        if c['width']>camera.width*.1 or c['height']>camera.height*.06 or ev[0]<1:continue
        p=camera.plane(c['center'],.1715)
        # Area must match the projection of a 26 mm opening, within occlusion tolerance.
        ring=np.array([p+[.013*np.cos(a),.013*np.sin(a),0] for a in np.linspace(0,2*np.pi,65)[:-1]])
        proj=camera.project(ring);expected=np.pi*np.ptp(proj[:,0])*np.ptp(proj[:,1])/4
        if not .40<c['area']/expected<1.25:continue
        choices.append((p,c))
    if len(choices)<4:raise ValueError('Four orange apertures not reliably visible')
    # Match the physical four-hole row using spacing, without a known world location.
    best=None
    from itertools import combinations
    for comb in combinations(choices,4):
        comb=sorted(comb,key=lambda x:x[0][0]);pts=np.array([x[0][:2] for x in comb])
        origin=np.median(pts-np.c_[np.arange(4)*.034,np.zeros(4)],axis=0)
        errors=np.linalg.norm(pts-(origin+np.c_[np.arange(4)*.034,np.zeros(4)]),axis=1)
        score=float(np.mean(errors))
        if best is None or score<best[0]:best=(score,origin,comb)
    if best[0]>.003:raise ValueError('Orange hole row geometry inconsistent')
    centers=best[1]+np.c_[np.arange(4)*.034,np.zeros(4)]
    return centers,{'row_fit_error_m':best[0],'detections':[x[1]['center'].tolist() for x in best[2]]}

_ANGLES=np.linspace(0,2*np.pi,32,endpoint=False)
def ellipse_residual(cam,point,u,v,radius,pixels,edge_offset):
    ring=point+radius*(np.cos(_ANGLES)[:,None]*u+np.sin(_ANGLES)[:,None]*v)
    projected=cam.project(ring);origin=projected.mean(0);scale=max(float(np.std(projected)),1.)
    x,y=((projected-origin)/scale).T
    design=np.c_[x*x,x*y,y*y,x,y];coeff=np.linalg.lstsq(design,np.ones(len(x)),rcond=None)[0]
    Q=np.array([[coeff[0],coeff[1]/2],[coeff[1]/2,coeff[2]]]);linear=coeff[3:]
    center=-.5*np.linalg.solve(Q,linear);denom=1+center@Q@center;Q=Q/denom
    uv=(pixels-origin)/scale-center
    rad=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',uv,Q,uv),1e-12))
    pixel_radius=scale/max(np.linalg.det(Q),1e-12)**.25
    return (rad-1)*pixel_radius+edge_offset

class TubeTracker:
    def __init__(self):self.last=None
    def estimate(self,images,cameras, predicted_center=None, predicted_axis=None, supported=False):
        caps=[liquid_surface(im) for im in images]
        for i,(c,cam) in enumerate(zip(caps,cameras)):
            if c is None:continue
            edge=c['boundary'];inside=(edge[:,0]>2)&(edge[:,0]<cam.width-2)&(edge[:,1]>2)&(edge[:,1]<cam.height-2)
            # A rack rim can cut through the cyan surface during insertion.
            # Pixels bordering that opaque rim are occlusion boundaries, not
            # points on the projected circular liquid surface.
            rgb=np.asarray(images[i],dtype=float);red,green,blue=rgb.transpose(2,0,1)
            rack=((red>green*1.22)&(green>blue*1.35)&(red>100)&(green>50)) | ((blue>green*1.3)&(blue>red*1.5)&(blue>100))
            near_rack=binary_dilation(rack,iterations=2)
            ij=np.floor(edge).astype(int)
            inside &= ~near_rack[ij[:,1],ij[:,0]]
            c['clipped']=not bool(inside.all());c['boundary']=edge[inside]
            if len(c['boundary'])<8:caps[i]=None
        valid=[i for i,c in enumerate(caps) if c is not None]
        if not valid:raise ValueError('Liquid surface not visible')
        if self.last is None:
            p=cameras[valid[0]].plane(caps[valid[0]]['center'],.143)
            initial=np.r_[p,0.,0.]
        else:initial=self.last.copy()
        if predicted_center is not None:
            axis=np.array([0.,0.,1.]) if predicted_axis is None else predicted_axis
            initial=np.r_[np.asarray(predicted_center)-axis*.018,axis[:2]/max(axis[2],.5)]
        prior=initial.copy()
        rejected_edges={}
        if predicted_center is not None:
            normal=np.r_[prior[3:5],1.];normal/=np.linalg.norm(normal)
            u=np.cross([0,1,0],normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
            for i in valid:
                cap=caps[i];cam=cameras[i]
                ring=prior[:3]+.008*(np.cos(_ANGLES)[:,None]*u+np.sin(_ANGLES)[:,None]*v)
                tolerance=max(2.5,float(np.std(cam.project(ring),axis=0).mean())*.08)
                distance=abs(ellipse_residual(cam,prior[:3],u,v,.008,cap['boundary'],.35))
                keep=distance<tolerance
                rejected_edges[str(i)]=int((~keep).sum())
                # The cyan side wall can join the top surface in one connected
                # component. Retain current image edges consistent with the
                # previously tracked circle, never the side-wall silhouette.
                if keep.sum()<8 or keep.mean()<.25:
                    caps[i]=None
                else:
                    cap['clipped']=cap['clipped'] or not bool(keep.all())
                    cap['boundary']=cap['boundary'][keep]
            valid=[i for i,c in enumerate(caps) if c is not None]
            if not valid:raise ValueError('No current image contour agrees with tracked surface')
        # Initialize translation by intersecting calibrated view rays when both are visible.
        if len(valid)==2 and not any(caps[i]['clipped'] for i in valid):
            A=[];B=[]
            for i in valid:
                v=cameras[i].ray(caps[i]['center']);M=np.eye(3)-np.outer(v,v);A.append(M);B.append(M@cameras[i].position)
            p=np.linalg.lstsq(np.vstack(A),np.concatenate(B),rcond=None)[0]
            if np.linalg.norm(p-initial[:3])<.12:initial[:3]=p
        def params(x):
            n=np.r_[x[3:5],1.];n/=np.linalg.norm(n)
            u=np.cross([0,1,0],n);u/=np.linalg.norm(u);v=np.cross(n,u)
            return n,u,v
        def residual(x):
            n,u,v=params(x);res=[]
            for i in valid:
                cam=cameras[i];cap=caps[i]
                res.extend(ellipse_residual(cam,x[:3],u,v,.008,cap['boundary'],.35))
            # Weak upright/continuity prior resolves ambiguous circle-normal solutions.
            res.extend((x[3:5]-prior[3:5])*8)
            if supported:res.append((x[2]-.143)/.0002)
            elif predicted_center is not None:res.append((x[2]-prior[2])/(.001 if len(valid)==1 else .2))
            return np.asarray(res)
        fit=least_squares(residual,initial,bounds=([.1,-.3,.09,-.6,-.6],[.75,.5,.4,.6,.6]),loss='soft_l1',f_scale=1.,max_nfev=35,diff_step=1e-4)
        n,_,_=params(fit.x);err=float(np.sqrt(np.mean(residual(fit.x)**2)))
        if err>2.5:raise ValueError('Liquid geometry fit unreliable')
        self.last=fit.x.copy()
        return fit.x[:3]+n*.018,n,{'fit_rms_pixels':err,'views':valid,'rejected_contour_pixels':rejected_edges,'cap_centers':[None if c is None else c['center'].tolist() for c in caps],'areas':[0 if c is None else c['area'] for c in caps]}
