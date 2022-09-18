import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import param_util
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c, G
import numpy as np  # efficient vector and matrix operations

# observation parameters:
background_rms = 0.5 # background noise per pixel
exp_time = 100 # exposure time (arbitrary units)
numPix = 100 # number of pixels
deltaPix = 0.05 # pixel size in arcsec
# PSF specification
fwhm = 0.1 # PSF FWHM
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix,
exp_time,
background_rms)
data_class = ImageData(**kwargs_data)
kwargs_psf = {'psf_type': 'GAUSSIAN','fwhm': fwhm,'pixel_size': deltaPix,'truncation': 5}
psf_class = PSF(**kwargs_psf)
# lens parameters
f=0.7
sigmav=200.
pa=np.pi/4.0 # position angle in radians
zl=0.3 # lens redshift
zs=1.5 # source redshift
# lens Einstein radius
co = FlatLambdaCDM(H0=70, Om0=0.3)
dl=co.angular_diameter_distance(zl)
ds=co.angular_diameter_distance(zs)
dls=co.angular_diameter_distance_z1z2(zl,zs)
# compute the Einstein radius
thetaE=1e6*(4.0*np.pi*sigmav**2/c**2*dls/ds*180.0/np.pi*3600.0).value
# eccentricity computation
e1,e2=(1-f)/(1+f)*np.cos(-2*pa),(1-f)/(1+f)*np.sin(-2*pa)
lens_model_list = ['SIE']
kwargs_sie = {'theta_E': thetaE,'center_x': 0,'center_y': 0,'e1': e1,'e2': e2}
kwargs_lens = [kwargs_sie]
lens_model_class = LensModel(lens_model_list=lens_model_list)