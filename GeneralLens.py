import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
import numpy as np  # efficient vector and matrix operations
from astropy.cosmology import FlatLambdaCDM
from Class_files import PSIEc, sersic

co = FlatLambdaCDM(H0=70, Om0=0.3)

# lens params
kwargs={'theta_c': 2.0,
        'norm': 10.0,
        'ell': 0.4,
        'zl': 0.5,
        'zs': 1.0}

el=PSIEc(co,size=80,npix=1000,**kwargs)

# size of the source image
size_stamp=150.0
npix_stamp=1000
xmin,xmax=-el.size/2,el.size/2
ymin,ymax=-el.size/2,el.size/2
fig,ax=plt.subplots(1,2,figsize=(14,8))

# sersic source with no lensing
beta=[0,0]
kwargs={'q': 0.5,
        're': 1.0,
        'pa': np.pi/4.0,
        'n': 1,
        'ys1': beta[0],
        'ys2': beta[1],
        'zs': 1.0}

se_unlensed=sersic(size_stamp,npix_stamp,**kwargs) # same source with lensing by the lens el
se=sersic(size_stamp,npix_stamp,gl=el,**kwargs)

# compute the time delay surface for a source at beta
td=el.t_delay_surf(beta=beta)

# draw caustics (on the left) and critical lines (on the right)
el.caustics(ax=ax[0],lt='--',alpha=1.0)
el.clines(ax=ax[1],lt='--',alpha=1.0)

# show unlensed (on the left) and lensed (on the right) images
ax[0].imshow(se_unlensed.image,origin='lower',
             extent=[-se.size/2,se.size/2,-se.size/2,se.size/2],
             cmap='gray_r')

ax[1].imshow(se.image,origin='lower',
             extent=[-se.size/2,se.size/2,-se.size/2,se.size/2],
             cmap='gray_r')

# show contours of the time delay surface
el.show_contours(td,ax=ax[1],minx=xmin,miny=ymin, nlevels=35,levmax=500,fontsize=20)
x0,x1=-20,20
y0,y1=-20,20
ax[0].set_xlim([x0,x1])
ax[0].set_ylim([y0,y1])
fig.tight_layout()