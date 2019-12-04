import numpy as np

from constants import GasConst

from utils import as_hdf5

_ln10 = np.log(10)

def create_eos_from_hdf5(f):
    types = [AdiabaticEoS]

    for t in types:
        try:
            return t.from_hdf5(f)
        except KeyError:
            pass
    else:
        raise ValueError("EOS object not found in file")

class IdealGasState(object):
    """Wrapper object for ideal gas properties at a P-T given state"""
    def __init__(self, P, T, gamma, mu):
        self._P = P
        self._T = T
        self._gamma = gamma
        self._mu = mu
        self._1 = np.ones_like(P)
    
    @property
    def pressure(self):
        return self._P
    @property
    def temperature(self):
        return self._T
    @property
    def density(self):
        return self._P * self._mu / (GasConst * self._T)

    @property
    def entropy(self):
        rho = self.density
        return self._P / rho**self._gamma
    @property
    def internal_energy(self):
        return GasConst * self._T / (self._mu * (self._gamma - 1))
    @property
    def dlne_dlnP(self):
        return 0.0*self._1
    @property
    def dlne_dlnT(self):
        return 1.0*self._1

    @property
    def Delta_ad(self):
        """Adiabatic gradient"""
        return (1 - 1 / self._gamma)*self._1
    @property
    def dDelta_dlnT(self):
        return 0*self._1
    @property
    def dDelta_dlnP(self):
        return 0*self._1

    @property
    def dlnrho_dlnT(self):
        """At constant pressure"""
        return -1.0*self._1
    @property
    def dlnrho_dlnP(self):
        """At constant temperature"""
        return  1.0*self._1

    @property
    def Cp(self):
        """Heat Capacity at constant pressure"""
        return self._1*self._gamma * GasConst / (self._gamma - 1)
    @property
    def dlnCp_dlnT(self):
        return 0*self._1
    @property
    def dlnCp_dlnP(self):
        return 0*self._1
    @property
    def Cv(self):
        """Heat Capacity at constant volume"""
        return self._1 * GasConst / (self._gamma - 1)
    @property
    def sound_speed_sqd(self):
        return self._gamma * self._P / self.density

    @property
    def vapor_pressure(self):
        return np.inf * np.ones_like(self._P)

class AdiabaticEoS(object):
    def __init__(self, gamma=1.4, mu=2.35):
        self._gamma = gamma
        self._mu = mu

    def to_hdf5(self, f):
        """Write to HDF5"""
        f = as_hdf5(f, 'w')

        eos = f.create_group(self.__class__.__name__)

        ds = eos.create_dataset('gamma', data=self._gamma)
        ds.attrs['unit'] = ''

        ds = eos.create_dataset('mu', data=self._mu)
        ds.attrs['unit'] = ''

    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')

        eos = f[cls.__name__]

        return cls(gamma=eos['gamma'].value, mu=eos['mu'].value)

    def get_state(self, P, T, derivs=False):
        return IdealGasState(P=P, T=T, gamma=self._gamma, mu=self._mu)

    def get_PT(self, rho=None, entropy=None, P=None, T=None, u=None):
        """Determine default variables, P and T, from a pair of 
        specified parameters"""
        if P is not None:
            if rho is None:
                rho = self.density(P, T=T, entropy=entropy)
            return P, self.temperature(rho, P=P)
        if T is not None:
            if rho is None:
                rho = (GasConst * T / (entropy*self._mu))**(1./ (self._gamma-1))
            return self.pressure(rho, T=T), T

    def pressure(self, rho, T=None, entropy=None, u=None):
        if T is not None:        
            return rho * GasConst * T / self._mu
        elif entropy is not None:
            return entropy * rho**self._gamma
        elif u is not None:
            return rho * u * (self._gamma - 1)
        else:
            raise ValueError("T, u, or entropy must be specified")
    
    def entropy(self, P, T):
        rho = self.density(P, T)
        return P / rho**self._gamma

    def density(self, P, T=None, entropy=None):
        if T is not None:
            return P * self._mu / (GasConst * T)
        elif entropy is not None:
            return (P/entropy)**(1/self._gamma)
        else:
            raise ValueError("T or entropy must be specified")

    def temperature(self, rho, entropy=None, P=None):
        if entropy is not None:
            P = self.pressure(rho, entropy=entropy)
        return (P / rho) * self._mu / GasConst
    
    def mu(self, P, T):
        return self._mu

