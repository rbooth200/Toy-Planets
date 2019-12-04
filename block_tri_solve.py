from __future__ import print_function

import numpy as np
import os
import ctypes as ct

from scipy.optimize import OptimizeResult

# Relies on the c++ library, which can be built with
#  g++ -O3 -shared  blck_tri_solve.cpp -I/usr/include/eigen3/ -Wall -fPIC -o blck_tri_solve.so -DEIGEN_NO_DEBUG


# Load the c++ library
DIR = os.path.dirname(os.path.realpath(__file__))
TRI_LIB = os.path.realpath(os.path.join(DIR, 'blck_tri_solve.so'))

libTriSolver = ct.CDLL(TRI_LIB)

tri_factor = libTriSolver.factor_block_tridiag
tri_factor.restype = ct.c_void_p

tri_solve = libTriSolver.solve_block_tridiag
tri_solve.restype = None

tri_delete = libTriSolver.delete_tridiag
tri_delete.restype = None

tri_factor_solve = libTriSolver.block_tridiag_solve
tri_factor_solve.restype = None


class BlockTriDiagSolver(object):
    """Solver for Tri-diagonal systems where each element is a matrix

    Parameters
    ----------
    tri_matrix : array, shape=[3,N,M,M]  
        Array of matrix coefficients:
            tri_matrix[0] - contains the N-1 MxM matrices on the lower
                            diagonal. tri_matrix[0,0] is not used.
            tri_matrix[1] - contains the N MxM matrices on the diagonal. 
            tri_matrix[2] - contains the N-1 MxM matrices on the upper 
                            diagonal. tri_matrix[2.,N-1] is not used.

    Notes
    -----
    Each of the NxN submatrices should be stored in C-order.
    """

    def __init__(self, tri_matrix):

        if len(tri_matrix.shape) != 4:
            raise ValueError("tri_matrix must be a a 4-d array")
        if tri_matrix.shape[0] != 3:
            raise ValueError("First dimension must be 3")
        if tri_matrix.shape[2] != tri_matrix.shape[3]:
            raise ValueError("sub-matrices must be square")

        # Get the matrix shape
        self._n_blocks = tri_matrix.shape[1]
        self._block_size = tri_matrix.shape[2]

        self._mat = tri_matrix

        # Flatten the sub-matrices:
        l, d, u = [np.array(t.flat, dtype='f8') for t in tri_matrix]

        self._solver = tri_factor(ct.c_int(self._n_blocks),
                                  ct.c_int(self._block_size),
                                  l.ctypes.data_as(ct.POINTER(ct.c_double)),
                                  d.ctypes.data_as(ct.POINTER(ct.c_double)),
                                  u.ctypes.data_as(ct.POINTER(ct.c_double)))

        self._solver = ct.c_void_p(self._solver)
        self._destroy = tri_delete

    def solve(self, rhs):
        """Solve the tri-diagonal system for the given parameter vector

        Parameters
        ----------
        rhs : array, size=n*m
            Right hand side to solve for.

        Returns
        -------
        x : array, size=n*m
            Solution
        """
        rhs = np.array(rhs, dtype='f8')
        if len(rhs) != self._n_blocks * self._block_size:
            raise ValueError("rhs must be a vector of length n*m")

        x = np.empty_like(rhs)

        tri_solve(self._solver,
                  rhs.ctypes.data_as(ct.POINTER(ct.c_double)),
                  x.ctypes.data_as(ct.POINTER(ct.c_double)))

        return x

    def as_dense_matrix(self):
        """Convert the block tridiagonal matrix to a dense matrix"""

        N = self._block_size*self._n_blocks

        dense = np.zeros([N, N])

        for k in range(self._n_blocks):
            for i in range(self._block_size):
                for j in range(self._block_size):

                    id = self._block_size*k + i
                    jd = self._block_size*k + j

                    jl = jd - self._block_size
                    ju = jd + self._block_size

                    dense[id, jd] = self._mat[1, k, i, j]
                    if k > 0:
                        dense[id, jl] = self._mat[0, k, i, j]
                    if k+1 < self._n_blocks:
                        dense[id, ju] = self._mat[2, k, i, j]

        return dense

    def dot(self, x):
        """Compute the dot product of the block-tri diagonal matrix with a
        vector.

        Parameters
        ----------
        x : array
            Vector to compute the dot product with.

        Returns
        -------
        b : array
            Dot product results, b = dot(M, x)
        """
        m = self._block_size
        N = self._n_blocks
        M = self._mat
        
        b = np.zeros([N,m])
        x = x.reshape(N,1,m)

        b[1:] += (M[0, 1:]*x[:-1]).sum(2)
        b[:] += (M[1, :,]*x[:]).sum(2)
        b[:-1] += (M[2, :-1]*x[1:]).sum(2)

        return b.reshape(N*m)

    def __del__(self):
        try:
            self._destroy(self._solver)
        except AttributeError:
            pass

class BaseLineSearch(object):
    """Base class for back-tracking line searches."""
    def __init__(self, limit_step=None):
        self.reduction = None
        
        if limit_step is None:
            self.limit_step = lambda p, _: p
        else:
            self.limit_step = limit_step


class LineSearch(BaseLineSearch):
    """Back-tracking line search for Newton's method.


    Notes
    -----
    This implementation is baeed on numerical recipes.
    """

    def __init__(self, armijo=1e-4, min_step_frac=0.1, limit_step=None):       
        super(LineSearch, self).__init__(limit_step)

        self._armijo = armijo
        self._l_min = min_step_frac

    def __call__(self, func, jac, x0, p, f0=None):
        """Find a good step using backtracking.

        Parameters
        ----------
        func : function,
            The function that we are trying to find the root of
        jac : Jacobian object,
            Must provide a "dot" method that returns the dot-product of the
            Jacobian with a vector.
        x0 : array
            Current best guess for the solution
        p : array
            Step direction.
        f0 : array, optional.
            Evaluation of func at x0, i.e. func(x0). If not provided then this
            will be evaluated internally.

        Returns
        -------
        x_new : array
            Recommended new point
        f_new : array
            func(x_new)
        nfev : int
            Number of function evaluations
        failed : bool
            Whether the line-search failed to produce a guess
        """
        def eval(x):
            fvec = func(x)
            f = 0.5*np.dot(fvec, fvec)
            return fvec, f

        nfev = 0
        if f0 is None:
            f0, cost = eval(x0)
            nfev += 1
        else:
            cost = 0.5*np.dot(f0, f0)

        # First make sure the step isn't trying to change x by too much
        p = self.limit_step(p, x0)

        # Compute the expected change in f due to the step p assuming f is
        # exactly described by its linear expansion about the root:
        delta_f = np.dot(f0, jac.dot(p))

        if delta_f > 0:
            raise ValueError("Round off in slope calculation")

        # Estimate the minimum step-size
        lam_min = np.max(np.abs(p) / np.maximum(np.abs(x0), 1.0))
        lam_min = np.finfo(f0.dtype).eps / lam_min

        # Start with the full newton step
        lam = 1.0
        cost_save = lam_save = None
        while True:
            if lam < lam_min:
                self.reduction = lam
                return x0, f0, nfev, True

            x_new = x0 + lam*p

            f_new, cost_new = eval(x_new)
            nfev += 1

            # Have we got an acceptable step?
            if cost_new <= (cost + self._armijo*lam*delta_f):
                self.reduction = lam
                return x_new, f_new, nfev, False

            # Try back tracking:
            if lam == 1.0:
                # First attempt, make a second order model of the cost 
                # against lam.
                lam_new  = - 0.5*delta_f / (cost_new - cost - delta_f)
            else:
                # Use a third order model
                r1 = (cost_new - cost - lam*delta_f)/(lam*lam)
                r2 = (cost_save - cost - lam_save*delta_f)/(lam_save*lam_save)
                
                a = (r1 - r2) / (lam - lam_save)
                b = (lam*r2-lam_save*r1) / (lam - lam_save)

                if a == 0:
                    lam_new = - 0.5*delta_f / b
                else:
                    d = b*b- 3*a*delta_f
                    if d < 0:
                        lam_new = 0.5*lam
                    elif b <= 0:
                        lam_new = (-b+np.sqrt(d))/(3*a)
                    else:
                        lam_new = -1 * delta_f / (b + np.sqrt(d))

                    lam_new = min(0.5*lam, lam_new)

            lam_save = lam
            cost_save = cost_new
            lam = max(lam_new, self._l_min*lam)    


class SimpleLineSearch(BaseLineSearch):
    """A simple back-tracking line-search based on scipy's solve_bvp"""
    def __init__(self, limit_step=None):
        super(SimpleLineSearch, self).__init__(limit_step)

    def __call__(self, func, jac, x0, p, f0=None):
        """Find a good step using backtracking.

        Parameters
        ----------
        func : function,
            The function that we are trying to find the root of
        jac : Jacobian object,
            Must provide a "dot" method that returns the dot-product of the
            Jacobian with a vector.
        x0 : array
            Current best guess for the solution
        p : array
            Step direction.
        f0 : array, optional.
            Evaluation of func at x0, i.e. func(x0). If not provided then this
            will be evaluated internally.

        Returns
        -------
        x_new : array
            Recommended new point
        f_new : array
            func(x_new)
        nfev : int
            Number of function evaluations
        failed : bool
            Whether the line-search failed to produce a guess
        """
        # Armijo constant: relative improvement criterion
        armijo = 0.01

        # Reduction factor
        reduction = 0.5

        # Number of trials before accepting the solution
        n_trial = 4

        p = self.limit_step(p, x0)

        cost = np.dot(p,p)

        nfev = 0
        if f0 is None:
            f0 = func(x0)
            nfev += 1

        alpha = 1.0
        for _ in range(n_trial):
            x_new = x0 + alpha * p

            fx = func(x_new)
            nfev += 1

            step_new = jac.solve(fx)
            cost_new = np.sum(step_new*step_new)

            if cost_new < (1 - 2*alpha*armijo) * cost:
                break
            else:
                alpha *= reduction

        self.reduction = alpha
        return x_new, fx, nfev, False
        

def solve_block_tridiag_sysem(f, jac, guess, tol=1e-5, ftol=None,
                              max_iter=200, max_jev=100,
                              line_search=LineSearch()):
    """Solve a non-linear system of equations with a block tridiagonal
    jacobian matrix.

    The equations are solved using a newton-iteration with an affine invariant
    back-tracking line search to aid convergence.

    Parameters
    ----------
    f : function,
        System of N equations with N variables to find the root, f(x) = 0, of.
    jac : function,
        Computes the block-tridiagonal jacobian. The jacobian must have the 
        shape=(3,N,m,m), where N is the number of equations and m is the size
        of the blocks. The lower, middle and upper diagonals are stored in the
        0th, 1st, and second rows of the jacobian. Note that jac[0,0] and 
        jac[2,N-1] are not used.
    guess : 1D array, size=N
        Iniital trial solution
    tol : float, default=1e-5
        Tolerence for convergence in x. Solution is accepted when the change in
        x, dx, obeys:
            np.max(np.abs(dx)) < tol.   
    ftol : float,
        Tolerence for convergence in f(x). Solution is accepted when:
            np.max(np.abs(f(x))) < ftol.   
        If not supplied ftol = tol is assumed.
    max_iter : int, default=200
        Maximum number of iterations used in the hunt for the solution.
    max_jev : int, default=100
        Maximum number of Jacobian evaluations allowed.
    """
    if ftol is None:
        ftol = tol

    njev = 0
    nfev = 0
    redo_jacobian = True
    x = guess
    fx = f(x)
    nfev += 1
    success = False
    fresh_jacobian = False
    for iter in range(1, max_iter+1):

        if redo_jacobian:
            J = jac(x)
            tri = BlockTriDiagSolver(J)
            njev += 1
            
            fresh_jacobian = True

        step = tri.solve(-fx)
        
        # Get the step size using a line-search method
        x_new, fx_new, nf, search_failed = line_search(f, tri, x, step, f0=fx)
        nfev += nf

        if search_failed:
            # Try again with a fresh jacobian
            redo_jacobian = True

            if fresh_jacobian:
                x = x + step
                fx =  f(x)
                nfev += 1
        else:
            x = x_new
            fx = fx_new

            # Test convergence
            #  Use a more-stringent criterion if we took a small step
            alpha = np.sqrt(line_search.reduction)
            if ((np.max(np.abs(fx))   < alpha*ftol) and 
                (np.max(np.abs(step)) < alpha*tol)):
                success = True
                break

        # If we took the maximum step we'll try again with the same jacobian
        if line_search.reduction == 1.0:
            redo_jacobian = False
            fresh_jacobian = False
        else:
            redo_jacobian = True
            if njev == max_jev:
                break

    if success:
        status = 0
        message = "The solution converged."
    elif iter == max_iter:
        status = 1
        message = ("Too many iterations were required. "
                   "Try increasing max_iter.")
    elif njev == max_jev:
        status = 2
        message = ("Too many jacobian evaluations were required. "
                   "Try increasing max_jev.")
    elif search_failed:
        status = 3
        message = ("No step found that improves the solution.")
    else:
        raise RuntimeError("Should never occur")

    return OptimizeResult(x=x, success=success, fun=fx, dx=step,
                          status=status, message=message,
                          nit=iter, nfev=nfev, njev=njev)


if __name__ == "__main__":

    mat = np.empty([3, 6, 4, 4])
    mat[0] = mat[2] = -np.eye(4).reshape(-1, 4, 4)
    mat[1] = 2*np.eye(4).reshape(-1, 4, 4)

    mat[0, 0] = np.nan
    mat[2, -1] = np.nan

    # Add a Random perturbation to make the matrices dense
    np.random.seed(42)
    mat += 1e-1*np.random.random(size=mat.shape)

    rhs = np.arange(np.prod(mat.shape[1:3]))

    tri = BlockTriDiagSolver(mat)
    x_tri = tri.solve(rhs)

    M = tri.as_dense_matrix()
    x_dens = np.linalg.solve(M, rhs)
    print('Tri-diag solver solution:\n', x_tri, '\n'
          'Dense solver solution:\n', x_dens)
    print('Max error:', np.max(np.abs(x_tri-x_dens)))

    # Check the matrix multiplication:
    print(tri.dot(x_tri))
    print(np.dot(M, x_tri))
