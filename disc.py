import numpy as np

from constants import GasConst, G

from utils import as_hdf5

def create_atmosphere_from_hdf5(f):
    types = [DiscAtmosphere, ]

    for t in types:
        try:
            return t.from_hdf5(f)
        except KeyError:
            pass
    else:
        raise ValueError("Atmosphere object not found in file")

class DiscAtmosphere(object):
    """Model for the planet atmosphere matching onto a circumstellar disc.
    
    Here the disc is modeled as being isothermal at temperature, T, with density,
    rho0, at large distances.
    args:
        rho0 : float
            Density at infinity, units = g cm^-3
        T : float
            Background temperature, units = K
        gamma : float
            Adiabatic index, default = 1.4
        mu : float
            Mean molecular weight, default = 2.35
    """
    def __init__(self, rho0, T, gamma=1.4, mu=2.35):
        self._rho0 = rho0
        self._T = T
        self._gamma = gamma
        self._mu = mu

    def to_hdf5(self, f):
        """Write to HDF5"""
        f = as_hdf5(f, 'w')

        atmo = f.create_group(self.__class__.__name__)

        ds = atmo.create_dataset('rho0', data=self._rho0)
        ds.attrs['unit'] = 'g cm^-3'

        ds = atmo.create_dataset('T', data=self._T)
        ds.attrs['unit'] = 'K'

        ds = atmo.create_dataset('gamma', data=self._gamma)
        ds.attrs['unit'] = ''

        ds = atmo.create_dataset('mu', data=self._mu)
        ds.attrs['unit'] = ''


    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')

        atmo = f[cls.__name__]

        return cls(atmo['rho0'].value, atmo['T'].value, 
                   gamma=atmo['gamma'].value, mu=atmo['mu'].value)

    def pressure(self, Mp, Rp):
        """External pressure given planet mass and radius"""
        return self.density(Mp, Rp) * GasConst * self._T / self._mu

    def temperature(self, Mp, Rp):
        """External temperature given planet mass and radius"""
        return self._T

    def density(self, Mp, Rp):
        """External temperature given planet mass and radius"""
        return self._rho0 * np.exp(self.Bondi_radius(Mp) / Rp)

    def Bondi_radius(self, Mp):
        cs2 = GasConst * self._T / self._mu 
        R_Bondi = G*Mp / cs2

        return R_Bondi
        
    def dPdr(self, Mp, R):
        return self.drhodr(Mp, R) * GasConst * self._T / self._mu
    def dTdr(self, Mp, R):
        return 0 
    def drhodr(self, Mp, R):
        Rb = self.Bondi_radius(Mp)
        return self._rho0 * np.exp(Rb/R) * (-Rb/R**2)

    @property
    def T0(self):
        return self._T
    @property
    def rho0(self):
        return self._rho0
    @property
    def P0(self):
        return self._rho0 * GasConst * self._T / self._mu

