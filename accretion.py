from __future__ import print_function
import abc
import numpy as np
import numba

from constants import G, Mearth


class Accretion(object):
    """Applies an accretion model to planet structure. 

    The class allows two types of accretion:
        1) Accretion onto the core
        2) Accretion onto the planet's outer layers.

    args:
        Tfac_atmo : float,
            Determines the region where atmospheric accretion is depositied,
            i.e. where the temperature is less than Tfac_atmo times larger than
            the temperature in the outermost cell. Default = 1.5
        verbose : int,
            Level of output, default = 0
    """
    def __init__(self, Tfac_atmo=1.1, verbose=0):
        self._Tfac_atmo = Tfac_atmo
        self._verbose = verbose

    def core_accretion(self, planet, dt):
        Mdot = self.core_accretion_rate(planet, dt)
        L_core = G*Mdot*planet.core.mass / planet.core.radius

        planet.set_core_accretion(Mdot, L_core)


    def atmospheric_accretion(self, planet, dt):
        """Apply accretion to the outer layers of the planet"""

        # Get the amount of mass to add to the atmosphere
        dM_atmo = self.atmos_accretion_rate(planet, dt)

        # Get the temperature at the middle of each cell
        me = planet.mass
        mc = 0.5*(me[1:]+me[:-1])
        T = planet.interpolate_structure(mc)['T']

        # Find the layer [idx, -1] such that all temperatures are
        # between T_ext and Tfac_atmo * T_ext
        T_ext = T[-1] * self._Tfac_atmo
        
        dT = T_ext - T
        idx = np.nonzero(dT[1:]*dT[:-1] < 0)[0]
        if len(idx):
            idx = idx[-1]+1
        else: # All temperatures are small enough
            idx = 0

        # Create a weighting function for the mass distribution:
        #    Weight more towards the outer edge
        Mf_atmo =  planet.shell_mass[idx:] / planet.shell_mass[idx:].sum()
        w = np.sqrt((Mf_atmo * T_ext / T[idx:]).cumsum())
        w /= w.sum()

        if self._verbose:
            print("Adding mass. Total fraction:", dM_atmo/planet.shell_mass.sum())
        if self._verbose > 1:
            print("Fractional changes per cell:\n", w*dM_atmo / planet.shell_mass[idx:])

        # Add/remove mass
        planet.shell_mass[idx:] += dM_atmo * w


    @abc.abstractmethod
    def core_accretion_rate(self, planet, dt):
        """Compute mass accretion onto the planet's core"""
        return

    @abc.abstractmethod
    def atmos_accretion_rate(self, planet, dt):
        """Compute mass accretion onto the planet's atmosphere in time dt"""
        return


class CompoundAccretion(object):
    """Multiple accretion objects"""
    def __init__(self, *accretors):
        self._accrete = accretors

    def __call__(self, planet, dt):
        for acc in self._accrete:
            acc(planet, dt)




class DiscAccretion(Accretion):
    """Handles accretion onto a planet from a disc"""
    def __init__(self, disc_atmo, **kwargs):
        super(DiscAccretion, self).__init__(**kwargs)

        self._disc = disc_atmo  

    def atmos_accretion_rate(self, planet, dt):
        """Add material to fill the planet's Bondi Radius"""

        # Work out the mass between the planet and the Bondi radius
        Mp = planet.total_mass 
        Rp = planet.radii[-1]

        Rb = self._disc.Bondi_radius(Mp)

        if self._verbose:
            print("Planet/Bondi radius:", Rp/Rb)

        disc = self._disc
        rho_bar = 0.5*(disc.density(Mp, Rb) + disc.density(Mp, Rp))
        M0 = 4 * np.pi * rho_bar * Rb**3 / 3.0

        x = Rp / Rb
        if x > 1:
            # Planet exceeds Bondi radius do not accrete
            return 0.0
        else:
            return M0 * (1 - x)

    def __call__(self, planet, dt):
        """Apply the accretion rate"""
        return self.atmospheric_accretion(planet, dt)

class DirectAccretion(Accretion):
    """Accretion of solids directly onto the core
    
    args:
        Mdot : function, Mdot(t)
            Accretion rate as a function of time
    """
    def __init__(self, Mdot):
        self._Mdot_s = Mdot

    def core_accretion_rate(self, planet, _):
        """Deposit solids inside the planet"""
        return self._Mdot_s(planet.age)

    def __call__(self, planet, dt):
        return self.core_accretion(planet, dt)
