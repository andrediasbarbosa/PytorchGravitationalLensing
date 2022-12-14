import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
import numpy as np  # efficient vector and matrix operations
from Class_files import point_bh, point_mass

bh=point_bh(3.0)
pm=point_mass(3.0)

r=np.linspace(3.0/2.0,10,1000)*2.0*bh.M
u=bh.u(r)/2.0/bh.M
a=bh.defAngle(r)
b=pm.defAngle(u*2.0*bh.M)
fig,ax=plt.subplots(1,1,figsize=(15,8))
ax.plot(u,a,'--',label='exact solution')
ax.plot(u,b,'--',label='weak field limit',color='red')
ax.set_xlabel(r'$u$ $[2GM/c^2]$')
ax.set_ylabel(r'$\hat\alpha(u)$ [radians]')
ax.legend()
x=[np.min(u),np.min(u)]
y=[0,10]
ax.plot(x,y,':')
plt.show()

