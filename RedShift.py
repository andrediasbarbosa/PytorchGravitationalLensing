from astropy.cosmology import FlatLambdaCDM
import numpy as np  # efficient vector and matrix operations
import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
from matplotlib.colors import SymLogNorm

gamma=1.0
kappa=1.0
lambdat=1.0-kappa-gamma
lambdar=1.0-kappa+gamma

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
zl=0.5
zs_norm=9.0
zs=np.linspace(zl,10.0,20)
dl=cosmo.angular_diameter_distance(zl)
ds=cosmo.angular_diameter_distance(zs)
dls=[]

for i in range(ds.size):
    dls.append(cosmo.angular_diameter_distance_z1z2(zl,zs[i]).value)

ds_norm=cosmo.angular_diameter_distance(zs_norm)
dls_norm=cosmo.angular_diameter_distance_z1z2(zl,zs_norm)
fig,ax=plt.subplots(1,2,figsize=(16,8))

ax[0].imshow(lambdat,origin='lower')
ax[1].imshow(lambdar,origin='lower')

for i in range(ds.size):
    kappa_new=kappa*ds_norm.value/dls_norm.value*dls[i]/ds[i].value
    gamma_new=gamma*ds_norm.value/dls_norm.value*dls[i]/ds[i].value
    lambdat_new=(1.0-kappa_new-gamma_new)
    lambdar_new=(1.0-kappa_new+gamma_new)
    ax[0].contour(lambdat_new,levels=[0.0])
    ax[1].contour(lambdar_new,levels=[0.0])

ax[0].contour(lambdat,levels=[0.0],colors="yellow",linewidths=2)
ax[1].contour(lambdar,levels=[0.0],colors="magenta",linewidths=2)
fig,ax=plt.subplots(1,2,figsize=(18,8))