"""
Functions and code snippets controlling the overall calculation, including
calling muSpectre.
"""
import numpy as np
import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction

from muTopOpt.Filter import map_to_unit_range
from muTopOpt.Filter import map_to_unit_range_derivative
from muTopOpt.PhaseField import phase_field_rectangular_grid_deriv_phase
from muTopOpt.PhaseField import phase_field_rectangular_grid
from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.StressTarget import square_error_target_stresses_deriv_strains
from muTopOpt.StressTarget import square_error_target_stresses_deriv_phase

def aim_function(phase, strains, stresses, cell, args):
    """ Calculate the aim function for the optimization.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    aim: float
         Value of aim function.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    aim = square_error_target_stresses(cell, strains, stresses, target_stresses)
    aim += weight * phase_field_rectangular_grid(phase, eta, cell)
    return aim

def aim_function_deriv_strains(phase, strains, stresses, cell, args):
    """ Calculate the partial derivative of the aim function with respect to the strains.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: list of np.ndarrays(dim**2 * nb_quad_pts * nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the strains.
    """
    target_stresses = args[0]
    derivatives = square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses)
    return derivatives

def aim_function_deriv_phase(phase, strains, stresses, cell, Young, delta_Young, Poisson,
                             delta_Poisson, dstress_dphase_list, args):
    """ Calculate the partial derivative of the aim function with respect to the design parameters.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    Young: np.ndarray(nb_pixels) of floats
           Youngs modulus for each pixel
    delta_Young: float
                 Max difference of Youngs modulus.
    Poisson: np.ndarray(nb_pixels) of floats
             Poissons ratio for each pixel
    delta_Poisson: float
                   Max difference of Poissons ratio.
    dstress_dphase: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the strains.
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: np.ndarray(nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the design parameters.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    derivatives = square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                                           dstress_dphase_list)
    derivatives *= map_to_unit_range_derivative(phase)
    derivatives += phase_field_rectangular_grid_deriv_phase(phase, eta, cell)
    return derivatives
