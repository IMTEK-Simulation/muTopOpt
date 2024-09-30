"""
This file contains the aim function and gradients for target stresses.
"""

import numpy as np
import muSpectre as µ

from NuMPI import MPI
from NuMPI.Tools import Reduction

def square_error_target_stresses(cell, strains, stresses, target_stresses, loading, Young,
                                 case_stress_entry=0):
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
    case_stress_entry: int
                       Wich entry of the stresses are compared?
                       1: Only xx-entry 2: Only yy-entry 3: Only xy- and yx-entry
                       Else (Default): All entries

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
        if case_stress_entry == 1:
            square_error += (stress_average[0, 0] - target_stress[0, 0])**2
        elif case_stress_entry == 2:
            square_error += (stress_average[1, 1] - target_stress[1, 1])**2
        elif case_stress_entry == 3:
            square_error += (stress_average[0, 1] - target_stress[0, 1])**2
            square_error += (stress_average[1, 0] - target_stress[1, 0])**2
        else:
            square_error += np.linalg.norm(stress_average - target_stress)**2

    return square_error


def square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses,
                                               loading, Young, case_stress_entry=0):
    """ Function to calculate the partial derivative of square_error_target_stresses
        with respect to the strains.

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
    case_stress_entry: int
                       Wich entry of the stresses are compared?
                       1: Only xx-entry 2: Only yy-entry 3: Only xy- and yx-entry
                       Else (Default): All entries

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
        if case_stress_entry == 1:
            derivative += 2 * (stress_average[0, 0] - target_stress[0, 0]) *\
                    dstress_dstrain[0, 0] / nb_pixels / cell.nb_quad_pts  / loading / Young
        elif case_stress_entry == 2:
            derivative += 2 * (stress_average[1, 1] - target_stress[1, 1]) *\
                    dstress_dstrain[1, 1] / nb_pixels / cell.nb_quad_pts  / loading / Young
        elif case_stress_entry == 3:
            derivative += 2 * (stress_average[0, 1] - target_stress[0, 1]) *\
                    dstress_dstrain[0, 1] / nb_pixels / cell.nb_quad_pts  / loading / Young
            derivative += 2 * (stress_average[1, 0] - target_stress[1, 0]) *\
                    dstress_dstrain[1, 0] / nb_pixels / cell.nb_quad_pts  / loading / Young
        else:
            for i in range(dim):
                for j in range(dim):
                    derivative += 2 * (stress_average[i, j] - target_stress[i, j]) *\
                        dstress_dstrain[i, j] / nb_pixels / cell.nb_quad_pts  / loading / Young
        derivative = derivative.flatten(order='F')
        derivatives.append(derivative)

    return derivatives

def square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                             dstress_dphase_list, loading, Young,
                                             case_stress_entry=0):
    """ Function to calculate the partial derivative of square_error_target_stresses
        with respect to the phase.

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
    case_stress_entry: int
                       Wich entry of the stresses are compared?
                       1: Only xx-entry 2: Only yy-entry 3: Only xy- and yx-entry
                       Else (Default): All entries

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
        if case_stress_entry == 1:
            derivative += 2 * (stress_average[0, 0] - target_stress[0, 0]) *\
                        dstress_dphase[0, 0] / nb_pixels
        elif case_stress_entry == 2:
            derivative += 2 * (stress_average[1, 1] - target_stress[1, 1]) *\
                        dstress_dphase[1, 1] / nb_pixels
        elif case_stress_entry == 3:
            derivative += 2 * (stress_average[0, 1] - target_stress[0, 1]) *\
                        dstress_dphase[0, 1] / nb_pixels
            derivative += 2 * (stress_average[1, 0] - target_stress[1, 0]) *\
                        dstress_dphase[1, 0] / nb_pixels
        else:
            for j in range(dim):
                for k in range(dim):
                    derivative += 2 * (stress_average[j, k] - target_stress[j, k]) *\
                        dstress_dphase[j, k] / nb_pixels

    derivative = derivative.flatten(order='F')
    return derivative
