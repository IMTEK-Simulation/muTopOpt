"""
This file contains the aim function and gradients for target stresses.
"""

import numpy as np
import muSpectre as µ

from NuMPI import MPI
from NuMPI.Tools import Reduction

def square_error_target_stresses(cell, strains, stresses, target_stresses, loading, Young):
    """ Function to calculate the square error between the average stresses and a list of target stresses.

    Parameters
    ----------
    cell: object
          muSpectre cell object
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    target_stresses: list of np.ndarray(dim, dim) of floats
                     List of target stresses. Must have the same length as stresses.
    loading: float
             Strain with which the stresses are nondimensionalized
    Young: float
           Youngs modulus with wich the stresses are nondimensionalized

    Returns:
    square_error: float
                  Square error between the average stresses and the target stresses.
    """
    dim = cell.dim
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]

    # Assert that stresses and target_stresses have the same length
    if len(stresses) is not len(target_stresses):
        exit('stresses and target_stresses must have the same length.')

    # Calculate square error
    square_error = 0
    for i in range(len(stresses)):
        # Aimed for average stress
        target_stress = target_stresses[i]
        # Actual average stress
        stress = stresses[i].reshape(shape, order='F')
        stress_average = np.empty((dim, dim))
        for j in range(dim):
            for k in range(dim):
                stress_average[j, k] = Reduction(MPI.COMM_WORLD).mean(stress[j, k])
        stress_average = stress_average / loading / Young
        # Square error
        square_error += np.linalg.norm(stress_average - target_stress)**2

    return square_error


def square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses, loading, Young):
    """ Function to calculate the partial derivative of square_error_target_stresses with respect to the strains.

    Parameters
    ----------
    cell: object
          muSpectre cell object
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    target_stresses: list of np.ndarray(dim, dim) of floats
                     List of target stresses. Must have the same length as strains.
    loading: float
             Strain with which the stresses are nondimensionalized
    Young: float
           Youngs modulus with wich the stresses are nondimensionalized

    Returns:
    derivatives: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
                 Partial derivative with respect to the strains.
    """
    dim = cell.nb_domain_grid_pts.dim
    nb_grid_pts = cell.nb_domain_grid_pts
    nb_pixels = np.prod(nb_grid_pts)
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]

    # Assert that strains and target_stresses have the same length
    if len(strains) is not len(target_stresses):
        exit('strains and target_stresses must have the same length.')

    # Calculate derivatives
    derivatives = []
    for k in range(len(strains)):
        target_stress = target_stresses[k]
        stress, dstress_dstrain = cell.evaluate_stress_tangent(strains[k].reshape(shape, order='F'))
        stress_average = np.empty((dim, dim))
        for i in range(dim):
            for j in range(dim):
                stress_average[i, j] = Reduction(MPI.COMM_WORLD).mean(stress[i, j])
        stress_average = stress_average / loading / Young
        derivative = 0
        for i in range(dim):
            for j in range(dim):
                derivative += 2 * (stress_average[i, j] - target_stress[i, j]) *\
                    dstress_dstrain[i, j] / nb_pixels / cell.nb_quad_pts  / loading / Young
        derivative = derivative.flatten(order='F')
        derivatives.append(derivative)

    return derivatives

def square_error_target_stresses_deriv_phase(cell, stresses, target_stresses, dstress_dphase_list, loading, Young):
    """ Function to calculate the partial derivative of square_error_target_stresses with respect to the phase.

    Parameters
    ----------
    cell: object
        muSpectre cell object
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of microscopic stresses. Must have the same length as target_stresses.
    target_stresses: list of np.ndarray(dim, dim) of floats
        List of target stresses. Must have the same length as stresses.
    dstress_dphase: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the strains.
    loading: float
             Strain with which the stresses are nondimensionalized
    Young: float
           Youngs modulus with wich the stresses are nondimensionalized

    Returns:
    derivative: np.ndarray(nb_pixels) of floats
                Partial derivative with respect to the phase.
    """
    dim = cell.nb_domain_grid_pts.dim
    lengths = cell.domain_lengths
    nb_grid_pts = cell.nb_domain_grid_pts
    nb_pixels = np.prod(nb_grid_pts)
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]

    # Assert that stresses and target_stresses have the same length
    if len(stresses) is not len(target_stresses):
        exit('stresses and target_stresses must have the same length.')

    # Calculate derivatives
    derivative = 0
    for i in range(len(stresses)):
        target_stress = target_stresses[i]
        stress = stresses[i].reshape(shape, order='F')
        stress_average = np.empty((dim, dim))
        for j in range(dim):
            for k in range(dim):
                stress_average[j, k] = Reduction(MPI.COMM_WORLD).mean(stress[j, k])
        stress_average = stress_average / loading / Young
        dstress_dphase = np.average(dstress_dphase_list[i], axis=2) / loading / Young
        for j in range(dim):
            for k in range(dim):
                derivative += 2 * (stress_average[j, k] - target_stress[j, k]) *\
                              dstress_dphase[j, k] / nb_pixels

    derivative = derivative.flatten(order='F')
    return derivative
