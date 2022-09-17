import numpy as np  # efficient vector and matrix operations
import scipy.special as sy

class point_bh:

    def __init__(self,M):
        self.M=M

    # functions defining the metric.
    def A(self,r):
        return(1.0-2.0*self.M/r)

    def B(self,r):
        return (self.A(r)**(-1))

    def C(self,r):
        return(r**2)

    # compute u from rm
    def u(self,r):
        u=np.sqrt(self.C(r)/self.A(r))
        return(u)

    # functions concurring to the deflection angle calculation
    def ss(self,r):
        return(np.sqrt((r-2.0*self.M)*(r+6.0*self.M)))

    def mm(self,r,s):
        return((s-r+6.0*self.M)/2/s)

    def phif(self,r,s):
        return(np.arcsin(np.sqrt(2.0*s/(3.0*r-6.0*self.M+s))))

    # the deflection angle
    def defAngle(self,r):
        s=self.ss(r)
        m=self.mm(r,s)
        phi=self.phif(r,s)
        F=sy.ellipkinc(phi, m) # using the ellipkinc function
        # from scipy.special
        return(-np.pi+4.0*np.sqrt(r/s)*F)


class point_mass:

    def __init__(self,M):
        self.M=M

# the classical formula
    def defAngle(self,u):
        return(4.0*self.M/u)