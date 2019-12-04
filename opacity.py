import numpy as np

from utils import as_hdf5

def create_opacity_from_hdf5(f):
    types = [DustOpacity, PowerLawOpacity]

    for t in types:
        try:
            return t.from_hdf5(f)
        except KeyError:
            pass
    else:
        raise ValueError("Atmosphere object not found in file")

class PowerLawOpacity(object):
    """Power-law model for the opacity,

    Kappa = k0 * rho**a * T**b

    Parameters
    ----------
    k0 : float
       Normalization
    a : float
        Power-law index on density
    b : float 
        Power-law index on temperature
    """
    def __init__(self, k0, a, b):
        self.k0 = k0
        self.a = a
        self.b = b

    def __call__(self, rho, T):
        """Compute the opacity"""
        return self.k0 * rho**self.a * T**self.b

    def dlogrho(self, rho, T):
        """ d(log(K)) / dlog(rho) at constant T"""
        return self.a
    
    def dlogT(self, rho, T):
        """ d(log(K)) / dlog(T) at constant rho"""
        return self.b


    def to_hdf5(self, f):
        """Write to HDF5"""
        f = as_hdf5(f, 'w')

        kap = f.create_group(self.__class__.__name__)

        kap.create_dataset('k0', data=self.k0)
        kap.create_dataset('a', data=self.k0)
        kap.create_dataset('a', data=self.k0)

    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')

        kap = f[cls.__name__]

        return cls(kap['k0'][()], kap['a'][()], kap['b'][()])

class DustOpacity(PowerLawOpacity):
    """Dust Opacity from Bell & Lin (1994)"""
    def __init__(self):
        super(DustOpacity, self).__init__(2e-4, 0.0, 2.0)

    def to_hdf5(self, f):
        """Write to HDF5"""
        f = as_hdf5(f, 'w')

        kap = f.create_group(self.__class__.__name__)

    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')

        kap = f[cls.__name__]

        return cls()
