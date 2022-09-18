from astropy.cosmology import FlatLambdaCDM
import numpy as np  # efficient vector and matrix operations
import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
from Class_files import sie_lens

# define a SIE lens with sigmav=200 km/s, f=0.3, pa=0.0
sigmav=200.0
f=0.3
pa=0.0
co = FlatLambdaCDM(Om0=0.3,H0=70.0)
sie=sie_lens(co,sigmav=sigmav,f=f,pa=pa)
# source coordinates
y1=0.2
y2=0.2
# set up a figure
fig,ax=plt.subplots(1,2,figsize=(18,8))
# polar coordinates of the images
x,phi=sie.phi_ima(y1,y2)
# convert to Cartesian coordinates
x1_ima=x*np.cos(phi)
x2_ima=x*np.sin(phi)
# compute cut, caustic, and critical line
y1_cut,y2_cut=sie.cut()
y1_cau,y2_cau=sie.tan_caustic()
x1_cc,x2_cc=sie.tan_cc()