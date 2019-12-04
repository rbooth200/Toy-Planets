from __future__ import print_function

import numpy as np
from scipy.interpolate import PchipInterpolator

from constants import Mearth, Rearth, m_H, GasConst, G, sig_SB, year

from EOS import create_eos_from_hdf5
from opacity import create_opacity_from_hdf5
from disc import create_atmosphere_from_hdf5

from block_tri_solve import (
    solve_block_tridiag_sysem, BlockTriDiagSolver, LineSearch
)
from utils import as_hdf5


def generate_structure_adiabatic(planet, rho0, P0, num_cells, gamma=1.4):
    """Create an initial structure with an adiabatic initial guess"""

    # Load saved variables
    eos = planet.eos
    core = planet.core

    # Compute the density / entropy at the Bondi radius:
    r_Bondi = G * core.mass / (P0 / rho0)
    rho_Bondi = rho0 * np.exp(1)
    P_Bondi = P0 * rho_Bondi / rho0
    T_Bondi = eos.temperature(rho_Bondi, P_Bondi)
    entr = eos.entropy(P_Bondi, T_Bondi)

    # To generate the structure we proceed as follows:
    #   - Set up an adiabatic structure neglecting envelope self-gravity
    #   - Then iterate to hydrostatic equilibrium

    r_e = np.geomspace(core.radius, r_Bondi, num_cells+1)
    r_c = 0.5*(r_e[1:] + r_e[:-1])

    # Compute the initial pressure, density, and shell masses
    P_c = rho_Bondi * (1 + (1-1/gamma) * (r_Bondi /
                                          r_c - 1.0))**(gamma/(gamma-1.))
    P_c, T_c = eos.get_PT(P=P_c, entropy=entr)
    rho_c = eos.density(P_c, T_c)

    dm_k = rho_c * 4 * np.pi * np.diff(r_e**3) / 3.

    # Setup the variables for the hydrostatic solver:
    L = np.ones_like(r_c)*planet.core.luminosity
    V = np.zeros_like(r_c)
    planet.set_structure(dm_k, r_e[1:], V, P_c, T_c, L)

    planet.solve_structure()


def generate_structure_isothernal(planet, rho0, P0, num_cells):
    """Create an initial structure with an isothermal initial guess"""

    # Load saved variables
    eos = planet.eos
    core = planet.core

    # Compute the density/Temperature at the Bondi radius:
    r_Bondi = G * core.mass / (P0 / rho0)
    rho_Bondi = rho0 * np.exp(1)

    # To generate the structure we proceed as follows:
    #   - Set up an adiabatic structure neglecting envelope self-gravity
    #   - Then iterate to hydrostatic equilibrium

    r_e = np.geomspace(core.radius, r_Bondi, num_cells+1)
    r_c = 0.5*(r_e[1:] + r_e[:-1])

    # Compute the initial pressure, density, and shell masses
    rho_c = rho_Bondi * np.exp(r_Bondi / r_c - 1.0)
    P_c = P0 * rho_c/rho0

    dm_k = rho_c * 4 * np.pi * np.diff(r_e**3) / 3.

    # Setup the variables for the hydrostatic solver:
    L = np.ones_like(r_c)*planet.core.luminosity
    V = np.zeros_like(r_c)
    planet.set_structure(dm_k, r_e[1:], V, P_c,
                         eos.temperature(rho_c, P=P_c), L)

    planet.solve_structure()


class PlanetCore(object):
    """Simple core of fixed mass

    args:
        Mc : float
           Core mass, units = earth masses
    """

    def __init__(self, Mp, Mdot=0, L0=0, age=0):

        self._M = Mp * Mearth

        self._Mdot = Mdot
        self._L0 = L0
        self._L = L0

        self._age = age

    def to_hdf5(self, f):
        """Save to HDF5"""

        f = as_hdf5(f, 'w')

        c = f.create_group(self.__class__.__name__)

        d = c.create_dataset('M', data=self._M/Mearth)
        d.attrs['unit'] = 'Mearth'

        d = c.create_dataset('Mdot', data=self._Mdot)
        d.attrs['unit'] = 'g s^-1'

        d = c.create_dataset('L0', data=self._L0)
        d.attrs['unit'] = 'erg s^-1'

        d = c.create_dataset('age', data=self._age)
        d.attrs['unit'] = 's'

        try:
            d = c.create_dataset('L', data=self.L)
            d.attrs['unit'] = 'erg s^-1'
        except AttributeError:
            pass

    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')
        c = f[cls.__name__]

        obj = cls(c['M'].value,
                  Mdot=c['Mdot'].value, L0=c['L0'].value, age=c['age'].value)

        try:
            obj._L = c['L'].value
        except KeyError:
            pass

        return obj

    @property
    def mass(self):
        return self._M

    @property
    def radius(self):
        return Rearth * (self._M/Mearth)**0.25

    @property
    def luminosity(self):
        return self._L

    @property
    def m_dot(self):
        return self._Mdot

    def set_age(self, age):
        self._M += self._Mdot*(age-self._age)
        self._age = age

    def set_core_luminosity(self, L0):
        self._L += L0 - self._L0
        self._L0 = L0

    def set_accretion(self, Mdot, L=None):
        self._Mdot = Mdot

        if L is not None:
            self._L = self._L0 + L


def create_core_from_hdf5(f):
    types = [PlanetCore, ]

    for t in types:
        try:
            return t.from_hdf5(f)
        except KeyError:
            pass
    else:
        raise ValueError("Core object not found in file")


class Planet(object):
    def __init__(self, core, atmosphere, eos, kappa, age=0, tol=1e-3):
        self._core = core
        self._atmo = atmosphere
        self._eos = eos
        self._kappa = kappa

        self._age = age

        self._tol = tol

        self._fC_last = None
        self._dt_last = None
        self._dt_next = None

    def to_hdf5(self, f):
        """Save to HDF5"""

        f = as_hdf5(f, 'w')

        pl = f.create_group(self.__class__.__name__)

        self._core.to_hdf5(pl)

        self._atmo.to_hdf5(pl)
        self._eos.to_hdf5(pl)
        self._kappa.to_hdf5(pl)

        # Save the structure data
        env = pl.create_group('envelope')
        env.create_dataset('mass', data=self._dm)

        R, V, P, T, L = self._R, self._V,self._P, self._T, self._L

        env.create_dataset('R', data=R[1:])
        env.create_dataset('V', data=V[1:])
        env.create_dataset('P', data=P)
        env.create_dataset('T', data=T)
        env.create_dataset('L', data=L[1:])

        # Extra kwargs
        kw = pl.create_group('kwargs')

        d = kw.create_dataset('age', data=self._age)
        d.attrs['unit'] = 's'
        d = kw.create_dataset('tol', data=self._tol)
        d.attrs['unit'] = ''

        # Auxillary variables:
        if self._fC_last is not None:
            aux = pl.create_group('aux')
            aux.create_dataset('fC_last', data=self._fC_last)
            aux.create_dataset('dt_last', data=self._dt_last)
            aux.create_dataset('dt_next', data=self._dt_next)

    @classmethod
    def from_hdf5(cls, f):
        """Load from HDF5 file"""
        f = as_hdf5(f, 'r')

        pl = f[cls.__name__]

        core = create_core_from_hdf5(pl)
        atmo = create_atmosphere_from_hdf5(pl)
        eos = create_eos_from_hdf5(pl)
        kappa = create_opacity_from_hdf5(pl)

        kw = pl['kwargs']
        self = cls(core, atmo, eos, kappa,
                   age=kw['age'].value, tol=kw['tol'].value)

        env = pl['envelope']
        dm = env['mass'].value

        RVPTL = [env[k].value for k in 'RVPTL']

        self.set_structure(dm, *RVPTL)

        # Load the extra variables
        try:
            aux = pl['aux']
            self._fC_last = aux['fC_last'].value
            self._dt_last = aux['dt_last'].value
            self._dt_next = aux['dt_next'].value
        except KeyError:
            pass

        return self

    def set_structure(self, dm, R, V, P, T, L):
        """Set the internal structure os the envelope.
        The envelope contains N cells, determined from the
        length of the shell mass array, dm.

        Parameters
        ----------
        dm : 1D array, len(dm)=M
            The mass contained in each shell
        R : 1D array, len(r_e)=N
            The outer edge radius of each shell.
        P : 1D array, len(P)=N
            Pressure at the cell centre
        T :  1D array, len(T)=N
            Temperature at the cell centre
        L : 1D array, len(L)=N
            Luminosity at the outer edge of each shell.
        """
        N = len(dm)+1
        self._R = np.empty(N)
        self._V = np.empty(N)
        self._L = np.empty(N)

        self._R[0] = self._core.radius
        self._R[1:] = R

        self._V[0] = 0
        self._V[1:] = V

        self._L[0] = self._core.luminosity
        self._L[1:] = L

        self._P = P
        self._T = T

        self._dm = dm
        self._mass = np.zeros(N)
        self._mass[1:] = dm.cumsum()

        # Setup interpolants
        m_e = self._mass
        m_c = 0.5*(m_e[1:]+m_e[:-1])
        self._interpRL = PchipInterpolator(
            m_e, [self._R, self._V, self._L], axis=1)
        self._interpPT = PchipInterpolator(
            m_c, [self._P, self._T], axis=1, extrapolate=True)

    def get_structure(self):
        """Get all components needed to recreate the envelope structure.

        This function can be used to save the current planet structure:
            saved = planet.get_structure()
            ** Make some unwise changes to the planet **
            # These changes can be un-done:
            planet.set_structure(*saved).
        """
        struct = [
            self._dm.copy(),
            self._R[1:].copy(),
            self._V[1:].copy(),
            self._P.copy(),
            self._T.copy(),
            self._L[1:].copy(),
        ]
        return struct

    def set_pressure(self, P):
        self._P

    def set_temperature(self, T):
        self._T

    def set_radii(self, R):
        self._R[1:] = R

    def set_core_accretion(self, Mdot_c, L_c):
        # Update luminosity to include change in core.
        self._L += L_c - self.core.luminosity

        self.core.set_accretion(Mdot_c, L_c)

    def solve_structure(self, dt=0, verbose=False):
        """Update the planet's envelope structure"""

        # Cache constants:
        core = self._core
        rc_0 =  core.radius


        core.set_age(self.age + dt)
        if dt:
            V_core = (core.radius - rc_0) / dt
        else:
            V_core = 0
        
        atmo = self._atmo
        eos = self._eos
        kappa = self._kappa

        # Normalization constants:
        R0 = self._R[-1]
        V0 = np.max(np.abs(self._V))
        P0 = self._P[0]
        T0 = self._T[0]
        L0 = self._L[-1]
        if L0 == 0:
            L0 = max(core.luminosity, 1)

        if dt:
            V0 = max(V0, abs(V_core))
        else:
            V0 = max(V0, 1)
            
        # Previous values of pressure / temperature
        if dt:
            rn = self._R / R0
            vn = self._V / V0
            pn = self._P / P0
            tn = self._T / T0

        # Enclosed mass at cell edges.
        m_e = core.mass + self._mass
        m_tot = m_e[-1]
        # Enclosed mass at cell centres:
        m_c = np.empty_like(m_e)
        m_c[:-1] = 0.5*(m_e[1:]+m_e[:-1])
        m_c[-1] = m_e[-1]

        # Mass difference between cell edges
        dm_e = self._dm
        # Mass difference between cell centres
        dm_c = np.empty_like(dm_e)
        dm_c[:-1] = 0.5*(dm_e[1:]+dm_e[:-1])
        dm_c[-1] = 0.5*dm_e[-1]

        k0 = 3*L0 / (64*np.pi*sig_SB*G*m_e[1:])

        kR = 3*dm_e/(4*np.pi*R0**3)
        kP = G*m_e[1:]*dm_c / (4*np.pi*R0**4 * P0)
        kP1 = dm_c * V0 / (4*np.pi*R0**2*P0)

        kL = dm_e/L0

        # Finite-difference form of the structure equations
        def struct_eqn(data, state=None):
            r = data[0::5]
            v = data[1::5]
            p = data[2::5]
            t = data[3::5]
            l = data[4::5]

            # Get the B.Cs
            r_core = core.radius / R0
            v_core = V_core / V0
            l_core = core.luminosity / L0
            p_surf = atmo.pressure(m_tot, R0*r[-1]) / P0
            t_surf = atmo.temperature(m_tot, R0*r[-1]) / T0

            P = p*P0
            T = t*T0

            Pbar = 0.5*(P[1:] + P[:-1])
            Tbar = 0.5*(T[1:] + T[:-1])

            if state is None:
                state = eos.get_state(P, T)

            Delta_ad = state.Delta_ad

            kap = kappa(state.density, T)
            kap = 0.5*(kap[1:]+kap[:-1])

            Delta_rad = k0 * l[1:] * kap * Pbar/Tbar**4

            Delta = np.minimum(0.5*(Delta_ad[1:]+Delta_ad[:-1]), Delta_rad)

            B = Delta * Tbar/Pbar * (P0/T0)

            r_eqn = np.empty_like(r)
            r_eqn[0] = r[0]**3 - r_core**3
            r_eqn[1:] = np.diff(r**3) - kR / state.density[:-1]

            v_eqn = np.zeros_like(v)
            if dt:
                v_eqn[0] = (v[0] - v_core)
                v_eqn[1:] = v[1:] - (r[1:] - rn[1:]) * R0 / (V0*dt)     

            p_eqn = np.empty_like(p)
            p_eqn[:-1] = np.diff(p) + kP/r[1:]**4
            p_eqn[-1] = p_surf - p[-1]

            if dt:
                p_eqn[:-1] += (kP1 / r[1:]**2) * (v[1:] - vn[1:]) / dt

            t_eqn = np.empty_like(t)
            t_eqn[:-1] = np.diff(t) + (kP/r[1:]**4) * B
            t_eqn[-1] = t_surf - t[-1]

            l_eqn = np.empty_like(l)
            l_eqn[0] = l[0] - l_core
            l_eqn[1:] = np.diff(l)

            if dt:
                dlnPdt = np.log(p[:-1]/pn) / dt
                dlnTdt = np.log(t[:-1]/tn) / dt
                dsdt = state.Cp[:-1] * (dlnTdt - Delta_ad[:-1] * dlnPdt)

                l_eqn[1:] += kL * T[:-1] * dsdt

            f = np.empty_like(data)
            f[0::5] = r_eqn
            f[1::5] = v_eqn
            f[2::5] = p_eqn
            f[3::5] = t_eqn
            f[4::5] = l_eqn

            return f

        # Tri-diagonal matrix equations
        def struct_jac(data):
            r = data[0::5]
            v = data[1::5]
            p = data[2::5]
            t = data[3::5]
            l = data[4::5]

            P = p*P0
            T = t*T0

            pbar = 0.5*(p[1:] + p[:-1])
            tbar = 0.5*(t[1:] + t[:-1])
            Pbar = pbar * P0
            Tbar = tbar * T0

            state = eos.get_state(P, T, derivs=True)

            dlrho_dlnP = state.dlnrho_dlnP
            dlrho_dlnT = state.dlnrho_dlnT

            jac = np.zeros([3, len(data)/5, 5, 5])

            # Radius
            kR_d = (kR / state.density[:-1])

            jac[0, 1:, 0, 0] = -3*r[:-1]**2
            jac[1,  :, 0, 0] = +3*r**2
            jac[0, 1:, 0, 2] = + kR_d * dlrho_dlnP[:-1] / p[:-1]
            jac[0, 1:, 0, 3] = + kR_d * dlrho_dlnT[:-1] / t[:-1]

            # Velocity
            jac[1, :, 1, 1] = 1.0
            if dt:
                jac[1, 1:, 1, 0] = -R0 / (V0*dt)

            # Presssure
            kP_r = (kP/r[1:]**4)
            jac[1, :-1, 2, 2] = -1

            jac[2, :-1, 2, 0] = -4*(kP_r/r[1:])
            jac[2, :-1, 2, 2] = +1

            jac[1, -1, 2, 0] = atmo.dPdr(m_tot, R0*r[-1]) * R0 / P0
            jac[1, -1, 2, 2] = -1

            if dt:
                jac[2, :-1, 2, 0] -= 2 * (kP1/r[1:]**3) * (v[1:] - vn[1:]) / dt
                jac[2, :-1, 2, 1] +=     (kP1/r[1:]**2)  / dt

            # Temperature
            # Derivatives of Delta / B
            Delta_ad = state.Delta_ad
            dDa_dP = state.dDelta_dlnP
            dDa_dT = state.dDelta_dlnT

            kap = kappa(state.density, T)
            kap = 0.5*(kap[1:]+kap[:-1])

            Delta_rad_l = k0 * kap * Pbar/Tbar**4
            Delta_rad = Delta_rad_l * l[1:]

            Delta = np.minimum(0.5*(Delta_ad[1:]+Delta_ad[:-1]), Delta_rad)

            dlogK_dlogrho = kappa.dlogrho(state.density, T)
            dlogK_dlogT = kappa.dlogT(state.density, T)

            dlogK_dlogP = dlogK_dlogrho*dlrho_dlnP
            dlogK_dlogT = dlogK_dlogrho*dlrho_dlnT + dlogK_dlogT

            dDr_dPp = 0.5*Delta_rad*(1. + dlogK_dlogP)[1:]
            dDr_dPm = 0.5*Delta_rad*(1. + dlogK_dlogP)[:-1]
            dDr_dTp = 0.5*Delta_rad*(-4. + dlogK_dlogT)[1:]
            dDr_dTm = 0.5*Delta_rad*(-4. + dlogK_dlogT)[:-1]

            dDr_dlp = Delta_rad_l

            dD_dPp = np.where(Delta_rad == Delta, dDr_dPp, dDa_dP[1:])
            dD_dPm = np.where(Delta_rad == Delta, dDr_dPm, dDa_dP[:-1])
            dD_dTp = np.where(Delta_rad == Delta, dDr_dTp, dDa_dT[1:])
            dD_dTm = np.where(Delta_rad == Delta, dDr_dTm, dDa_dT[:-1])

            dD_dlp = np.where(Delta_rad == Delta, dDr_dlp, 0.0)

            B = Delta * tbar/pbar

            dB_dpp = -0.5*B/pbar + dD_dPp * tbar / pbar**2
            dB_dpm = -0.5*B/pbar + dD_dPm * tbar / pbar**2

            dB_dtp = +0.5*Delta / pbar + dD_dTp * 1/pbar
            dB_dtm = +0.5*Delta / pbar + dD_dTm * 1/pbar

            dB_dlp = dD_dlp * tbar/pbar

            # Now for the jacobian components
            jac[1, :-1, 3, 2] = +    kP_r * dB_dpm
            jac[1, :-1, 3, 3] = -1 + kP_r * dB_dtm

            jac[2, :-1, 3, 0] = - 4*(kP_r/r[1:]) * B
            jac[2, :-1, 3, 2] = +    kP_r * dB_dpp
            jac[2, :-1, 3, 3] = +1 + kP_r * dB_dtp
            jac[2, :-1, 3, 4] = +    kP_r * dB_dlp

            # Temperature boundary
            jac[1, -1, 3, 0] = atmo.dTdr(m_tot, R0*r[-1]) * R0 / T0
            jac[1, -1, 3, 3] = -1

            # Luminosity
            jac[0, :, 4, 4] = -1
            jac[1, :, 4, 4] = +1

            if dt:
                Cp = state.Cp[:-1]
                dlCp_dlT = state.dlnCp_dlnT[:-1]
                dlCp_dlP = state.dlnCp_dlnP[:-1]

                term = Cp*T[:-1]*kL / dt
                dlnP = np.log(p[:-1]/pn)
                dlnT = np.log(t[:-1]/tn)

                fL = term * (dlnT - Delta_ad[:-1] * dlnP)

                jac[0, 1:, 4, 2] += fL*(dlCp_dlP)/p[:-1]
                jac[0, 1:, 4, 2] -= (term/p[:-1]) * \
                    (Delta_ad[:-1] + dDa_dP[:-1]*dlnP)
                jac[0, 1:, 4, 3] += fL*(dlCp_dlT + 1)/t[:-1]
                jac[0, 1:, 4, 3] += (term/t[:-1])*(1 - dDa_dT[:-1]*dlnP)

            return jac

        # Set-up extended arrays including boundary conditions:
        N = self.size+1
        r, v, p, t, l = [np.empty(N) for i in range(5)]

        r[:] = self._R / R0
        v[:] = self._V / V0
        p[:-1] = self._P / P0
        t[:-1] = self._T / T0
        l[:] = self._L / L0

        # Apply boundaries
        r[0] = core.radius / R0
        l[0] = core.luminosity / L0
        v[0] = V_core / V0
        p[-1] = atmo.pressure(m_tot, R0*r[-1]) / P0
        t[-1] = atmo.temperature(m_tot, R0*r[-1]) / T0

        # Create a set of initial guesses - using the old structure.
        guess = np.empty(5*N)
        guess[0::5] = r
        guess[1::5] = v
        guess[2::5] = p
        guess[3::5] = t
        guess[4::5] = l

        # Prevent p, t from going below zero
        def limit(p, x):
            p = p.reshape(-1, 5)
            x = x.reshape(-1, 5)

            alpha = min(np.min(np.where(p[:, 2] < 0,
                                        -0.25*x[:, 2]/p[:, 2], 1.0)),
                        np.min(np.where(p[:, 3] < 0,
                                        -0.25*x[:, 3]/p[:, 3], 1.0)))
            alpha = min(alpha, 1.0)
            return alpha*p.reshape(-1)

        search = LineSearch(limit_step=limit)

        try:
            sol = solve_block_tridiag_sysem(
                struct_eqn, struct_jac, guess, max_iter=1000, line_search=search)
        except FloatingPointError:
            print('Error in structure equations!\n'
                  'Saving input structure to dump.h5')
            self.to_hdf5('dump.h5')
            raise
        
        # If the radii become disordered, we'll need to try again.
        if not np.all(np.diff(sol.x[0::5]) > 0):
            return False

        if not sol.success:
            err_max = np.max(np.abs(sol.fun))
            if err_max < 1e-4:
                print("Accepting lower tolerance:\n\t", err_max)
                sol.success = True

        if sol.success:
            # Save the solution

            R = sol.x[0::5][1:] * R0
            V = sol.x[1::5][1:] * V0
            P = sol.x[2::5][:-1] * P0
            T = sol.x[3::5][:-1] * T0
            L = sol.x[4::5][1:] * L0

            self.set_structure(self._dm, R, V, P, T, L)
        else:
            print(sol.success, sol.message)
            print('Max. change in solution on last iteration:\n\t',
                  np.max(np.abs(sol.dx)))
            print('Max. error in structure equations (scaled):\n\t',
                  np.max(np.abs(sol.fun)))

        return sol.success

    def evolve(self, tmax, remesher=None, accretion=None):
        """Evolve the planetary structure to a new age, tmax"""

        def compute_change(old, new):
            def err_rel(x1, x2):
                """Compute the r.m.s relative error"""
                return np.sqrt((((x1-x2)/np.maximum(x1, x2))**2).mean())

            err_r = err_rel(new[1], old[1])
            #err_V = err_rel(new[2], old[2])
            err_P = err_rel(new[3], old[3])
            err_T = err_rel(new[4], old[4])
            err_L = err_rel(new[5], old[5])

            err_M = err_rel(new[0], old[0])

            err = (err_r + err_P + err_T + err_L)/4. + 3*err_M
            return self._tol / err

        def switch(x):
            return 1 + 2*np.arctan(0.5*(x-1))

        # Use an initial guess of 0.1% of KH time
        if self._dt_next:
            dt = self._dt_next
        elif self.luminosity[-1] > 0:
            dt = self.KelvinHelmholtzTime() / 1000
        else:
            dt = 3600.0*24  # Use 1 day

        f0 = self._fC_last
        dt0 = self._dt_last
        while self._age < tmax:
            if remesher:
                remesher(self)
            save = self.get_structure()

            while True:
                accretion(self, dt)

                if self.solve_structure(dt=dt):
                    break  # Step completed
                else:
                    print('Failed, using shorter dt:',  0.5*dt/year)
                    self.set_structure(*save)
                    dt /= 2

            new = self.get_structure()
            fi = switch(compute_change(save, new))
            if dt0:
                dt_next = dt * switch(fi*f0/switch(dt/dt0))**0.25
            else:
                dt_next = dt

            self._age += dt
            dt0 = dt
            f0 = fi
            dt = dt_next

        self._dt_last = dt0
        self._dt_next = dt
        self._fC_last = f0

    def adapt_mesh(self, splits, merges, assume_sorted=True, verbose=1):
        """Update the mesh according to the specified splits and merges.

        Parameters
        ----------
        splits : list
            List of cell splits. Each split is a tuple containing the index
            of the cell to split and the number of cells to divide it into.
        merges : list
            List of cells to merge. Each merge is a tuple specifying the index
            of first celll to merge and the number of following cells.
        verbose: int, default= 1
            Level of information to print to screen.
        """
        if assume_sorted == False:
            raise ValueError("Non-sorted splits and merges not currently "
                             "supported")

        # Return if there is no work to do
        if len(splits) == 0 and len(merges) == 0:
            return

        if verbose:
            print("Remeshing: time {:.3e}. {} splits and {} merges. {} cells"
                  "".format(self.age/year, len(splits), len(merges),
                            len(self.radii)))

        # Get the structure arrays:
        dm = self.shell_mass.copy()
        me = self.mass.copy()

        r = self.radii.copy()
        v = self.velocity.copy()
        P = self.pressure.copy()
        T = self.temperature.copy()
        L = self.luminosity.copy()

        def _insert(array, loc, data, axis=0):
            return np.insert(array, [loc+1], data, axis=axis)

        # Generate the new cells via splitting
        for s, n in reversed(splits):
            if verbose > 1:
                print('Splitting cell {} into {}'.format(s, n))

            # Compute the new edges and centres in mass-space
            #   Note: The new cell centres may not include the old one
            dm_s = dm[s] / n

            i = np.arange(0, n+1)
            mi = me[s] + dm_s*i

            mc = 0.5*(mi[1:]+mi[:-1])
            mi = mi[1:-1]  # We can use the old start/finish cell

            # Update the mass-grid
            me = _insert(me, s, mi)

            dm[s] = dm_s
            dm = _insert(dm, s, dm_s*np.ones_like(mi))

            # Update the cell edge variables
            ri, vi, Li = self._interpRL(mi)
            r = _insert(r, s, ri)
            v = _insert(v, s, vi)
            L = _insert(L, s, Li)

            # Now the cell centre variables:
            Pi, Ti = self._interpPT(mc)
            P[s] = Pi[0]
            P = _insert(P, s, Pi[1:])
            T[s] = Ti[0]
            T = _insert(T, s, Ti[1:])

            # Update merges list to correct for new structure
            if len(merges > 0):
                merges[merges[:, 0] > s, 0] += n - 1

        # Now remove cells that are not needed:
        for s, n in reversed(merges):
            if verbose > 1:
                print('Merging cells {} to {}'.format(s, s+n))

            i = s + np.arange(1, n)

            dm[s] = np.sum(dm[s:s+n])

            # Delete the old cells from the extended arrays
            me = np.delete(me, i)

            # Delete the old cells from the structure arrays
            dm = np.delete(dm, i)
            r = np.delete(r, i)
            v = np.delete(v, i)
            P = np.delete(P, i)
            T = np.delete(T, i)
            L = np.delete(L, i)

            # For the cell-centred quantities, compute the new states
            mc = 0.5*(me[s]+me[s+1])
            P[s], T[s] = self._interpPT(mc)

        # Now that we have the new structure, save it.
        self.set_structure(dm, r[1:], v[1:], P, T, L[1:])

        # That's all folks

    def interpolate_structure(self, m):
        """Interpolate the structure equations"""
        rvl = self._interpRL(m)
        pt = self._interpPT(m)
        return { 
            'R' : rvl[0],
            'V' : rvl[1],
            'L' : rvl[2],
            'P' : pt[0],
            'T' : pt[1]
        }

    @property
    def size(self):
        return self._P.shape[0]

    @property
    def age(self):
        return self._age

    @property
    def eos(self):
        return self._eos

    @property
    def opacity(self):
        return self._kappa

    @property
    def atmosphere(self):
        return self._atmo

    @property
    def core(self):
        return self._core

    @property
    def mass(self):
        return self._mass

    @property
    def shell_mass(self):
        return self._dm

    @property
    def total_mass(self):
        return self._core.mass + self._mass[-1]

    @property
    def radii(self):
        return self._R

    @property
    def velocity(self):
        return self._V

    @property
    def pressure(self):
        return self._P

    @property
    def temperature(self):
        return self._T

    @property
    def luminosity(self):
        return self._L

    def get_shell_volume(self):
        """Volume of each shell"""
        rs = self.radii
        return 4*np.pi*np.diff(rs**3) / 3.

    def get_density(self):
        """Density in each shell"""
        return self._eos.density(self.pressure, T=self.temperature)

    def KelvinHelmholtzTime(self):
        """Cooling time"""
        M_k = self._core.mass + self.shell_mass.cumsum()
        re = self.radii
        rc = 2 / (1/re[1:] + 1/re[:-1])
        U = (G*M_k*self.shell_mass / rc).sum()
        L = self.luminosity[-1]

        return U / L


if __name__ == "__main__":
    # Try to initialize the structure
    from EOS import AdiabaticEoS
    from opacity import DustOpacity
    from disc import DiscAtmosphere
    from accretion import DiscAccretion, DirectAccretion, CompoundAccretion
    from mesh import Remesher

    import matplotlib.pyplot as plt
    np.seterr(invalid='raise')

    class TaperedAccretion(object):
        """Tapered accretion rate profile"""

        def __init__(self, M0, Mmax, tmax, age=0):
            self._dM = (Mmax - M0) * Mearth
            self._tmax = tmax

        def __call__(self, t):
            x = t / self._tmax
            x = np.minimum(x, 1)
            return (self._dM / self._tmax) * (2 - 6*x**2 + 4*x**3)
        
    # Grow to core 2 MEarth in 10^5 years
    core = PlanetCore(0.01, L0=0)
    Mdot = TaperedAccretion(0.01, 2, 1e5*year)

    eos = AdiabaticEoS()


    rho0, T0 = 2.5e-9, 300.0
    P, T = eos.get_PT(rho=rho0, T=T0)
    mu = eos.mu(P, T)

    disc = DiscAtmosphere(rho0=rho0, T=T0, mu=mu)


    accretion1 = DiscAccretion(disc)
    accretion2 = DirectAccretion(Mdot)
    accretion = CompoundAccretion(accretion1, accretion2)

    core.set_accretion(Mdot(0), G*core.mass*Mdot(0) / core.radius)

    kappa = DustOpacity()

    pl = Planet(core, disc, eos, kappa, tol=1e-2)
    mesher = Remesher(verbose=1)

    generate_structure_isothernal(pl, disc.rho0, disc.P0, 500)

    r_e = pl.radii / 1e10
    r_c = 0.5*(r_e[1:]+r_e[:-1])

    ax1 = plt.subplot(511)
    ax1.loglog(r_e, pl.luminosity/3.83e33)
    ax1.set_ylabel('luminosity (Lsun)')

    ax2 = plt.subplot(512, sharex=ax1)
    ax2.loglog(r_c, pl.pressure / disc.P0, label=str(pl.age/year)+'yr')
    ax2.set_ylabel('P [dyn]')

    ax3 = plt.subplot(513, sharex=ax1)
    ax3.loglog(r_c, pl.get_density())
    ax3.set_ylabel('rho [g/cm^3]')

    ax4 = plt.subplot(514, sharex=ax1)
    ax4.loglog(r_c, pl.temperature)
    ax4.set_ylabel('T [K]')


    ax5 = plt.subplot(515, sharex=ax1)
    ax5.semilogx(r_e, pl.velocity)
    ax5.set_ylabel('v [cm/s]')

    ax5.set_xlabel('R [10^10 cm]')
    plt.subplots_adjust(hspace=0)

    if pl.age == 0:
        pl.to_hdf5('output/{}yr.h5'.format(0/year))

    for t in [1, 10, 100, 1e3, 1e4, 1e5, 2e5, 3e5, 4e5, 5e5, 1e6, 2e6, 3e6, 5e6, 1e7]:

        t *= year

        if t <= pl.age:
            continue

        pl.evolve(t, mesher, accretion)
        pl.to_hdf5('output/{}yr.h5'.format(round(pl.age/year, 0)))

        r_e = pl.radii / 1e10
        r_c = 0.5*(r_e[1:]+r_e[:-1])

        print("Time: ", pl.age/year, 'yr')
        print("Core Mass (Mearth):", pl.core.mass/Mearth)
        print("Envelope Mass (Mearth):", pl.shell_mass.sum()/Mearth)
        print("K-H time (yr):", pl.KelvinHelmholtzTime()/year)
        if Mdot(pl.age) > 0:
            print("Mass fraction reaching core:", pl.core._Mdot/Mdot(pl.age))
        print("Number of cells:", pl.radii.size)

        ax1.loglog(r_e, pl.luminosity/3.83e33)
        ax2.loglog(r_c, pl.pressure / disc.P0, label=str(t/year)+'yr')
        ax3.loglog(r_c, pl.get_density())
        ax4.loglog(r_c, pl.temperature)
        ax5.semilogx(r_e, pl.velocity)

    plt.show()
