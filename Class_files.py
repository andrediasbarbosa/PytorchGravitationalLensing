import matplotlib.pyplot as plt  # a MATLAB-like plotting framework
import numpy as np  # efficient vector and matrix operations
from scipy.ndimage import map_coordinates
import astropy.io.fits as pyfits
import numpy.fft as fftengine
import scipy.special as sy


class point_bh:

    def __init__(self, M):
        self.M = M

    # functions defining the metric.
    def A(self, r):
        return (1.0 - 2.0 * self.M / r)

    def B(self, r):
        return (self.A(r) ** (-1))

    def C(self, r):
        return (r ** 2)

    # compute u from rm
    def u(self, r):
        u = np.sqrt(self.C(r) / self.A(r))
        return (u)

    # functions concurring to the deflection angle calculation
    def ss(self, r):
        return (np.sqrt((r - 2.0 * self.M) * (r + 6.0 * self.M)))

    def mm(self, r, s):
        return ((s - r + 6.0 * self.M) / 2 / s)

    def phif(self, r, s):
        return (np.arcsin(np.sqrt(2.0 * s / (3.0 * r - 6.0 * self.M + s))))

    # the deflection angle
    def defAngle(self, r):
        s = self.ss(r)
        m = self.mm(r, s)
        phi = self.phif(r, s)
        F = sy.ellipkinc(phi, m)  # using the ellipkinc function
        # from scipy.special
        return (-np.pi + 4.0 * np.sqrt(r / s) * F)


class point_mass:

    def __init__(self, M):
        self.M = M

    # the classical formula
    def defAngle(self, u):
        return (4.0 * self.M / u)


class sersic(object):

    def __init__(self, size, N, gl=None, **kwargs):

        if ('n' in kwargs):
            self.n = kwargs['n']
        else:
            self.n = 4

        if ('re' in kwargs):
            self.re = kwargs['re']
        else:
            self.re = 5.0

        if ('q' in kwargs):
            self.q = kwargs['q']
        else:
            self.q = 1.0

        if ('pa' in kwargs):
            self.pa = kwargs['pa']
        else:
            self.pa = 0.0

        if ('ys1' in kwargs):
            self.ys1 = kwargs['ys1']
        else:
            self.ys1 = 0.0

        if ('ys2' in kwargs):
            self.ys2 = kwargs['ys2']
        else:
            self.ys2 = 0.0

        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0

        self.N = N
        self.size = float(size)
        self.df = gl

        # define the pixel coordinates
        pc = np.linspace(-self.size / 2.0, self.size / 2.0, self.N)
        self.x1, self.x2 = np.meshgrid(pc, pc)
        if self.df != None:
            ds_lens = gl.ds
            dls_lens = gl.dls
            ds = gl.co.angular_diameter_distance(self.zs)
            dls = gl.co.angular_diameter_distance_z1z2(gl.zl, self.zs)
            self.corrf = ds_lens / dls_lens * dls / ds
            if self.zs != gl.zs:
                gl.rescale(self.corrf)
            y1, y2 = self.ray_trace()
        else:
            y1, y2 = self.x1, self.x2

        self.image = self.brightness(y1, y2)

    def brightness(self, y1, y2):

        # rotate the galaxy by the angle self.pa
        x = np.cos(self.pa) * (y1 - self.ys1) + np.sin(self.pa) * (y2 - self.ys2)
        y = -np.sin(self.pa) * (y1 - self.ys1) + np.cos(self.pa) * (y2 - self.ys2)
        # include elliptical isophotes
        r = np.sqrt(((x) / self.q) ** 2 + (y) ** 2)
        # brightness at distance r
        bn = 1.992 * self.n - 0.3271
        brightness = np.exp(-bn * ((r / self.re) ** (1.0 / self.n) - 1.0))
        return (brightness)

    def ray_trace(self):
        """Ray-tracing through the lens place"""
        px = self.df.pixel
        x1pix = (self.x1 + self.df.size / 2.0) / px
        x2pix = (self.x2 + self.df.size / 2.0) / px
        # compute the deflection angles at the light ray positions
        # on the lens plane. Use the deflection angles of the
        # general lens self.df
        a1 = map_coordinates(self.df.a1, [x2pix, x1pix], order=2) * px
        a2 = map_coordinates(self.df.a2, [x2pix, x1pix], order=2) * px
        # apply the lens equation
        y1 = (self.x1 - a1)  # y1 coordinates on the source plane
        y2 = (self.x2 - a2)  # y2 coordinates on the source plane
        return (y1, y2)

    def rescale(self, fcorr):

        if self.pot_exists:
            self.pot = self.pot * fcorr
            self.a1 = self.a1 * fcorr
            self.a2 = self.a2 * fcorr
            self.a12 = self.a12 * fcorr
            self.a11 = self.a11 * fcorr
            self.a22 = self.a22 * fcorr
            self.a21 = self.a21 * fcorr


class gen_lens(object):
    """
    Initialize gen_lens
    """

    # the lens does not have a potential yet
    def __init__(self):
        self.pot_exists = False

    # convergence
    def convergence(self):
        if (self.pot_exists):
            kappa = 0.5 * (self.a11 + self.a22)
        else:
            print("The lens potential is not initialized yet")
        return (kappa)

    # shear
    def shear(self):
        if (self.pot_exists):
            g1 = 0.5 * (self.a11 - self.a22)
            g2 = self.a12
        else:
            print("The lens potential is not initialized yet")
        return (g1, g2)

    # determinant of the Jacobian matrix
    def detA(self):
        if (self.pot_exists):
            deta = (1.0 - self.a11) * (1.0 - self.a22) - self.a12 * self.a21
        else:
            print("The lens potential is not initialized yet")
        return (deta)

    # critical lines overlaid to the map of detA, returns a set of
    # contour objects
    def crit_lines(self, ax=None, show=True):
        if (ax == None):
            print("specify the axes to display the critical lines")
        else:
            deta = self.detA()
            # ax.imshow(deta,origin='lower')
            cs = ax.contour(deta, levels=[0.0], colors='white', alpha=0.0)
            if show == False:
                ax.clear()
        return (cs)

    # plot of the critical lines in the axes ax
    def clines(self, ax=None, color='red', alpha=1.0, lt='-'):

        cs = self.crit_lines(ax=ax, show=False)
        contour = cs.collections[0]
        p = contour.get_paths()
        sizevs = np.empty(len(p), dtype=int)
        no = self.pixel
        # if we found any contour, then we proceed
        if (sizevs.size > 0):
            for j in range(len(p)):
                # for each path, we create two vectors containing
                # the x1 and x2 coordinates of the vertices
                vs = contour.get_paths()[j].vertices
                sizevs[j] = len(vs)
                x1 = []
                x2 = []
                for i in range(len(vs)):
                    xx1, xx2 = vs[i]
                    x1.append(float(xx1))
                    x2.append(float(xx2))

                # plot the results!
                ax.plot((np.array(x1) - self.npix / 2.) * no, (np.array(x2) - self.npix / 2.) * no, lt, color=color,
                        alpha=alpha)

    # plot of the caustics in the axes ax
    def caustics(self, ax=None, alpha=1.0, color='red', lt='-'):
        cs = self.crit_lines(ax=ax, show=True)
        contour = cs.collections[0]
        p = contour.get_paths()  # p contains the paths of each individual critical line
        sizevs = np.empty(len(p), dtype=int)
        # if we found any contour, then we proceed
        if (sizevs.size > 0):
            for j in range(len(p)):
                # for each path, we create two vectors containing
                # the x1 and x2 coordinates of the vertices
                vs = contour.get_paths()[j].vertices
                sizevs[j] = len(vs)
                x1 = []
                x2 = []
                for i in range(len(vs)):
                    xx1, xx2 = vs[i]
                    x1.append(float(xx1))
                    x2.append(float(xx2))

                a_1 = map_coordinates(self.a1, [[x2], [x1]], order=1)
                a_2 = map_coordinates(self.a2, [[x2], [x1]], order=1)
                # now we can make the mapping using the lens equation:
                no = self.pixel
                y1 = (x1 - a_1[0] - self.npix / 2.) * no
                y2 = (x2 - a_2[0] - self.npix / 2.) * no
                # plot the results!
                ax.plot(y1, y2, lt, color=color, alpha=alpha)

    # geometrical time delay
    def t_geom_surf(self, beta=None):
        x = np.arange(0, self.npix, 1, float) * self.pixel
        y = x[:, np.newaxis]
        if beta is None:
            x0 = y0 = self.npix / 2 * self.pixel
        else:
            x0 = beta[0] + self.npix / 2 * self.pixel
            y0 = beta[1] + self.npix / 2 * self.pixel

        return 0.5 * ((x - x0) * (x - x0) + (y - y0) * (y - y0))

    # gravitational time delay:
    def t_grav_surf(self):
        return -self.pot

    # total time delay
    def t_delay_surf(self, beta=None):
        t_grav = self.t_grav_surf()
        t_geom = self.t_geom_surf(beta)
        td = (t_grav + t_geom)
        return (t_grav + t_geom)

    # display the time delay contours
    def show_contours(self, surf0, ax=None, minx=-25, miny=-25,
                      cmap=plt.get_cmap('Paired'),
                      linewidth=1, fontsize=20, nlevels=40, levmax=100, offz=0.0):
        if ax == None:
            print("specify the axes to display the contours")
        else:
            minx = minx
            maxx = -minx
            miny = miny
            maxy = -miny
            surf = surf0 - np.min(surf0)
            levels = np.linspace(np.min(surf), levmax, nlevels)
            ax.contour(surf, cmap=cmap, levels=levels,
                       # linewidth=linewidth,
                       extent=[-self.size / 2, self.size / 2, -self.size / 2, self.size / 2])
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.set_xlabel(r'$\theta_1$', fontsize=fontsize)
            ax.set_ylabel(r'$\theta_2$', fontsize=fontsize)
            ax.set_aspect('equal')


class PSIEc(gen_lens):

    def __init__(self, co, size=100.0, npix=200, **kwargs):

        # set the cosmological model
        self.co = co

        # core radius
        if ('theta_c' in kwargs):
            self.theta_c = kwargs['theta_c']
        else:
            self.theta_c = 0.0

        # ellipticity
        if ('ell' in kwargs):
            self.ell = kwargs['ell']
        else:
            ell = 0.0

        # normalization norm
        if ('norm' in kwargs):
            self.norm = kwargs['norm']
        else:
            self.norm = 1.0

        # lens redshift zl
        if ('zl' in kwargs):
            self.zl = kwargs['zl']
        else:
            self.zl = 0.5

        # source redshift zs
        if ('zs' in kwargs):
            self.zs = kwargs['zs']
        else:
            self.zs = 1.0

        # angular diameter distances
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)

        # size of the output image
        self.size = size

        # number of pixels
        self.npix = npix

        # pixel scale
        self.pixel = self.size / self.npix

        # calculate the lensing potential
        self.potential()

    def potential(self):

        x = np.arange(0, self.npix, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = self.npix / 2
        self.pot_exists = True
        #
        self.pot = np.sqrt(((x - x0) * self.pixel) ** 2 / (1 - self.ell) + ((y - y0) * self.pixel) ** 2 * (
                    1 - self.ell) + self.theta_c ** 2) * self.norm
        self.a2, self.a1 = np.gradient(self.pot / self.pixel ** 2)
        self.a12, self.a11 = np.gradient(self.a1)
        self.a22, self.a21 = np.gradient(self.a2)


class deflector(gen_lens):

    def __init__(self, co, filekappa, zl=0.5, zs=1.0,
                 pad=False, npix=200, size=100):

        # read input convergence map from fits file
        kappa, header = pyfits.getdata(filekappa, header=True)
        self.co = co
        self.zl = zl
        self.zs = zs
        # angular diameter distances
        self.dl = co.angular_diameter_distance(self.zl)
        self.ds = co.angular_diameter_distance(self.zs)
        self.dls = co.angular_diameter_distance_z1z2(self.zl, self.zs)
        # pixel scale and number of pixels of the input convergence map
        self.pixel_scale = header['CDELT2'] * 3600.0
        self.kappa = kappa
        self.nx = kappa.shape[0]
        self.ny = kappa.shape[1]
        # pixels and size of the output lensing maps
        # they can be different from the those of the
        # input convergence map
        self.npix = npix
        self.size = size
        self.pixel = float(self.size) / self.npix
        # use zero-padding to compute the lensing potential
        self.pad = pad
        if (pad):
            self.kpad()
        self.potential()

    def kpad(self):

        def padwithzeros(vector, pad_width, iaxis, kwargs):
            vector[:pad_width[0]] = 0
            vector[-pad_width[1]:] = 0
            return vector

        # use the pad method from numpy.lib to add zeros (padwithzeros)
        # in a frame with thickness self.kappa.shape[0]
        self.kappa = np.lib.pad(self.kappa, self.kappa.shape[0], padwithzeros)

    # calculate the potential by solving the poisson equation
    # the output potential map has the same size of the input
    # convergence map
    def potential_from_kappa(self):

        # define an array of wavenumbers (two components k1,k2)
        k = np.array(np.meshgrid(fftengine.fftfreq(self.kappa.shape[0]), fftengine.fftfreq(self.kappa.shape[1])))
        # Compute Laplace operator in Fourier space = -4*pi*k^2
        kk = k[0] ** 2 + k[1] ** 2
        kk[0, 0] = 1.0
        # FFT of the convergence
        kappa_ft = fftengine.fftn(self.kappa)
        # compute the FT of the potential
        kappa_ft *= - 1.0 / (kk * (2.0 * np.pi ** 2))
        kappa_ft[0, 0] = 0.0
        potential = fftengine.ifftn(kappa_ft)
        if self.pad:
            pot = self.mapCrop(potential.real)
        return pot

    def potential(self):

        no = self.pixel
        x_ = np.linspace(0, self.npix - 1, self.npix)
        y_ = np.linspace(0, self.npix - 1, self.npix)
        x, y = np.meshgrid(x_, y_)
        potential = self.potential_from_kappa()
        x0 = y0 = potential.shape[0] / 2 * self.pixel_scale - self.size / 2.0
        x = (x0 + x * no) / self.pixel_scale
        y = (y0 + y * no) / self.pixel_scale
        self.pot_exists = True
        pot = map_coordinates(potential, [y, x], order=1)
        self.pot = pot * self.pixel_scale ** 2 / no / no
        self.a2, self.a1 = np.gradient(self.pot)
        self.a12, self.a11 = np.gradient(self.a1)
        self.a22, self.a21 = np.gradient(self.a2)
        self.pot = pot * self.pixel_scale ** 2

    def mapCrop(self, mappa):

        xmin = int(self.kappa.shape[0] / 2 - self.nx / 2)
        ymin = int(self.kappa.shape[1] / 2 - self.ny / 2)
        xmax = int(xmin + self.nx)
        ymax = int(ymin + self.ny)
        mappa = mappa[xmin:xmax, ymin:ymax]
        return (mappa)
