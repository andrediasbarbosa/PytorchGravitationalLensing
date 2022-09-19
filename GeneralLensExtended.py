import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
import numpy as np  # efficient vector and matrix operations
from astropy.cosmology import FlatLambdaCDM
from Class_files import PSIEc, sersic, deflector

size=200.0
npix=500
# create instance of the deflector class
df=deflector('data/kappa_2.fits',co=0.5,zl=0.5, zs=9.0, pad=True,npix=npix,size=size)
# set the source position and Sersic parameters:
beta=[-30,8]
kwargs={'q': 0.5,
're': 1.0,
'pa': np.pi/4.0,
'n': 1,
'ys1': beta[0],
'ys2': beta[1],
'zs': 9.0}
#
xmin,xmax=-df.size/2,df.size/2
ymin,ymax=-df.size/2,df.size/2
fig,ax=plt.subplots(1,2,figsize=(14,8))
se_unlensed=sersic(size_stamp,npix_stamp,**kwargs)
se=sersic(size_stamp,npix_stamp,gl=df,**kwargs)
td=df.t_delay_surf(beta=beta)
df.caustics(ax=ax[0],lt='--',alpha=1.0)
df.clines(ax=ax[1],lt='--',alpha=1.0)
ax[0].imshow(se_unlensed.image,origin='lower',
extent=[-se_unlensed.size/2,se_unlensed.size/2,
-se_unlensed.size/2,se_unlensed.size/2],
cmap='gray_r')
ax[1].imshow(se.image,origin='lower',
extent=[-se.size/2,se.size/2,-se.size/2,se.size/2],
cmap='gray_r')
df.show_contours(td,ax=ax[1],minx=xmin,miny=ymin,nlevels=25,
levmax=1600,fontsize=20)
x0,x1=-40,10
y0,y1=-25,25
ax[0].set_xlim([x0,x1])
ax[0].set_ylim([y0,y1])
x0,x1=-80,50
y0,y1=-65,65
ax[1].set_xlim([x0,x1])
ax[1].set_ylim([y0,y1])
fig.tight_layout()