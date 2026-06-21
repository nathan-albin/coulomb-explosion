"""
Functions for Coulomb explosion simulation.
"""

import numpy as np
import numba
from scipy.integrate import solve_ivp

# tolerances for the numerical solver
RKRTOL = 1e-8
RKATOL = 1e-16

def convert_masses(masses):
    """
    Convert masses from amu to electron mass.
    """
    return masses*1822.89

@numba.jit
def forces(x, charges):
    """
    Computes the Coulombic forces among atoms given the positions.

    Args:
        charges (np.ndarray): An array of charges.
        x (np.ndarray): An Nx3 array of position vectors.
    """

    # count number of atoms
    n_atoms = len(charges)

    # initialize force vectors
    F = np.zeros((n_atoms,3))

    # loop over pairs of atoms
    for i in range(n_atoms-1):
        for j in range(i+1,n_atoms):

            # direction vector
            v = x[i,:] - x[j,:]

            # magnitude of the direction
            norm_v = np.linalg.norm(v)

            # contribution to atom i
            F[i,:] += charges[i]*charges[j]/norm_v**3*v

            # contribution to atom j (opposite direction)
            F[j,:] -= charges[i]*charges[j]/norm_v**3*v

    return F

@numba.jit
def rhs(t,state,masses,charges):
    """
    The RHS function for Coulomb mechanics.
    """
    n_atoms = len(charges)

    # extract get position information
    x = state[:n_atoms*3].reshape((n_atoms,3))

    # compute the forces
    F = forces(x, charges)

    # convert to a vector of accelerations
    accel = (F/masses[:,np.newaxis]).flatten()

    # construct state derivative
    d_state = np.empty(n_atoms*6)
    d_state[:n_atoms*3] = state[n_atoms*3:]
    d_state[3*n_atoms:] = accel
    return d_state


def simulate_explosion(x_init, v_init, masses, charges):
    """
    Simulates a Coulomb explosion.

    Args:
        x_init (numpy.ndarray):  The Nx3 array of initial positions.
        v_init (numpy.ndarray):  The Nx3 array of initial velocities.
        masses (numpy.ndarray):  The N-dimensional array of masses.
        charges (numpy.ndarray): The N-dimensional array of charges.

    Returns:
        x_final (numpy.ndarray): The Nx3 array of final positions.
        v_final (numpy.ndarray): The Nx3 array of final velocities.
    """

    n_atoms = len(charges)

    # set initial state
    init_state = np.r_[x_init.flatten(), v_init.flatten()]

    # solve
    sol = solve_ivp(rhs, (0,1e10), init_state, t_eval=(1e10,), rtol=RKRTOL, atol=RKATOL, args=(masses,charges))

    # extract final positions and velocities
    x_final = sol.y[:3*n_atoms,0].reshape((n_atoms,3))
    v_final = sol.y[3*n_atoms:,0].reshape((n_atoms,3))
    
    assert np.sum(forces(x_final, charges)**2) < 1e-12

    return x_final, v_final
