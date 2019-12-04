import numpy as np
from scipy.interpolate import PchipInterpolator as interp

from constants import year

class Remesher(object):
    r"""Adaptive mesh refinement controls for the planet properties.
    This class uses a mesh-function, defined below to make recommendations
    to the planet class on which cells to split or merge. This is done via the
    mesh-function:
        f = c_1 (m/M) - c_2 log(P),
    where c_1 and c_2 are the constants 1 / mass_frac and c_2 = 1 / dlogP,
    respectively. 
    
    The mesh-function is evaluated at the edges of each cell and the Remesher
    recommends splitting cells with \Delta f > 1 into as many cells as needed
    to bring Delta f < 1. Merges are recommended when the sum of \Delta f 
    across two cells would remain < 1 after the merge.

    Parameters
    -----------
    dlogP : float
        Maximum change in the log-pressure between cells, default = 0.05
    mass_frac : float
        Maximum change in mass-fraction between cells, default = 0.02
    verbose : int
        Verbosity level, default = 0
    """
    def __init__(self, dlogP=0.1, mass_frac=0.02, verbose=0):
        self._dlogP = dlogP
        self._mass_frac = mass_frac 
        self._verbose = verbose

    def mesh_function(self, planet):
        """Target function for mesh-refinement."""

        me = planet.mass

        logP = np.log(planet.interpolate_structure(me)['P'])
        fm = me / me[-1]

        return fm/self._mass_frac - logP/self._dlogP
 
    def _determine_splits_and_merges(self, planet):
        """Determines which cells to split and merge"""

        f = self.mesh_function(planet)
        
        f = np.diff(f)

        # Cells with f > 1 need splitting
        splits = np.nonzero(f > 1)[0]
        splits = [[s, int(f[s]+1)] for s in splits]

        # If the sum of the mesh-function across consecutative cells is < 1,
        # then merge cells.
        i = 0 ; N = len(f)
        merges = []
        while i < N-2:
            ftot = f[i] 
            if ftot > 1:
                i += 1 # splitting
            else:
                j = 1
                if f[i] + f[i+j] < 1:
                    j += 1
                if j > 1:
                    merges.append([i,j])
            
                i += j

        splits = np.array(splits)
        merges = np.array(merges)
        return splits, merges


    def __call__(self, planet):
        """Evaluate the mesh-function and recommend changes in mesh structure
        to the planet object.
        
        Paramters
        ---------
        planet : planet object
            Planet to adapt the mesh of.
        """
        # First determine whether we need to split / merge any cells
        splits, merges = self._determine_splits_and_merges(planet)
        # Adapt the planet's mesh
        planet.adapt_mesh(splits, merges, verbose=self._verbose)
