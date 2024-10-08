"""
This file contains the aim function and gradients for target stresses.
"""
import sys
import os

import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ

from NuMPI import MPI
from NuMPI.Tools import Reduction

def square_error_target_stresses_sequential(cell, strains, stresses, target_stresses):
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
        stress_average = np.average(stress, axis=(2, 3, 4))
        # Square error
        normalization = np.linalg.norm(target_stress) ** 2
        err = np.linalg.norm(stress_average - target_stress)**2
        square_error +=  err / normalization

    return square_error

def square_error_target_stresses(cell, strains, stresses, target_stresses):
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
        # Square error
        normalization = np.linalg.norm(target_stress) ** 2
        err = np.linalg.norm(stress_average - target_stress)**2
        square_error +=  err / normalization

    return square_error


def square_error_target_stresses_deriv_strains_sequential(cell, strains, stresses,
                                                          target_stresses):
    """ Function to calculate the partial derivative of
        square_error_target_stresses with respect to the strains.

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
        # Target stress
        target_stress = target_stresses[k]

        # Average stress
        stress, dstress_dstrain =\
            cell.evaluate_stress_tangent(strains[k].reshape(shape, order='F'))
        stress_average = np.average(stress, axis=(2, 3, 4))

        # Derivative
        derivative = 0
        normalization = np.linalg.norm(target_stress) ** 2
        for i in range(dim):
            for j in range(dim):
                derivative += 2 * (stress_average[i, j] - target_stress[i, j]) *\
                    dstress_dstrain[i, j] / nb_pixels / cell.nb_quad_pts  / normalization
        derivative = derivative.flatten(order='F')
        derivatives.append(derivative)

    return derivatives

def square_error_target_stresses_deriv_strains(cell, strains, stresses,
                                               target_stresses):
    """ Function to calculate the partial derivative of
        square_error_target_stresses with respect to the strains.

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
        # Target stress
        target_stress = target_stresses[k]

        # Average stress
        stress, dstress_dstrain = cell.evaluate_stress_tangent(strains[k].reshape(shape, order='F'))
        stress_average = np.empty((dim, dim))
        for i in range(dim):
            for j in range(dim):
                stress_average[i, j] = Reduction(MPI.COMM_WORLD).mean(stress[i, j])

        # Derivative
        derivative = 0
        normalization = np.linalg.norm(target_stress) ** 2
        for i in range(dim):
            for j in range(dim):
                derivative += 2 * (stress_average[i, j] - target_stress[i, j]) *\
                    dstress_dstrain[i, j] / nb_pixels / cell.nb_quad_pts  / normalization
        derivative = derivative.flatten(order='F')
        derivatives.append(derivative)

    return derivatives

def square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                             dstress_dmat_list):
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
    dstress_dmat: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the
        material density.

    Returns:
    derivative: np.ndarray(nb_quad_pts, *nb_grid_pts) of floats
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
        # Target stress
        target_stress = target_stresses[i]
        # Average stress
        stress = stresses[i].reshape(shape, order='F')
        stress_average = np.empty((dim, dim))
        for j in range(dim):
            for k in range(dim):
                stress_average[j, k] = Reduction(MPI.COMM_WORLD).mean(stress[j, k])
        dstress_dmat = dstress_dmat_list[i]
        normalization = np.linalg.norm(target_stress) ** 2
        for j in range(dim):
            for k in range(dim):
                derivative += 2 * (stress_average[j, k] - target_stress[j, k]) *\
                    dstress_dmat[j, k] / nb_pixels / cell.nb_quad_pts / normalization
    return derivative
