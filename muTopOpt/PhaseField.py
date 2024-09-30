"""
This file contains the phase-field function to regularize
an optimization problem and the derivation of the phase-field
function with respect to the phase-field.
"""

import numpy as np
import muSpectre as µ

from NuMPI import MPI
from NuMPI.Tools import Reduction

################################################################################
### ---------------- Phase on nodes: rectangular triangles ----------------- ###
################################################################################
def phase_field_gradient_sequential(phase, cell):
    """ Function to calculate the gradient part of the
        phase-field function on a regular 2D grid.
        As discretization rectangular triangles are used. The
        phase field is defined on the nodes.

    Parameters
    ----------
    phase: np.ndarray(nb_grid_pts) of floats
           Phase field function.
    cell: object
          muSpectre cell object

    Returns
    -------
    gradient_term: float
                   Gradient term of the phase-field energy
    """
    Lx, Ly = cell.domain_lengths
    hx = Lx / cell.nb_domain_grid_pts[0]
    hy = Ly / cell.nb_domain_grid_pts[1]

    gradient_norm = np.empty([cell.nb_quad_pts, *phase.shape])
    # Lower left triangle
    phase_dx = (np.roll(phase, -1, axis=0) - phase) / hx
    phase_dy = (np.roll(phase, -1, axis=1) - phase) / hy
    gradient_norm[0] = phase_dx ** 2 + phase_dy ** 2
    # Upper right triangle
    phase_dx = (np.roll(phase, [-1, -1], axis=(0, 1)) - np.roll(phase, -1, axis=1)) / hx
    phase_dy = (np.roll(phase, [-1, -1], axis=(0, 1)) - np.roll(phase, -1, axis=0)) / hy
    gradient_norm[1] = phase_dx ** 2 + phase_dy ** 2

    # Integral
    gradient_term = np.average(gradient_norm) * Lx * Ly

    return gradient_term

def phase_field_double_well_potential_sequential(phase, cell):
    """ Function to calculate the double-well potential part of the
        phase-field energy, e.g. int(rho² - 2rho³ + rho⁴)dx on a regular 2D grid.
        The phase field is defined on the nodes with linear finite elements
        (rectangular triangles).

    Parameters
    ----------
    phase: np.ndarray(nb_grid_pts) of floats
           Phase field function.
    cell: object
          muSpectre cell object

    Returns
    -------
    gradient_term: float
                   Gradient term of the phase-field energy
    """
    # Determinante of Jacobian matrix
    det = np.prod(cell.domain_lengths) / cell.nb_pixels
    # Phase at the corners of each element
    phase_0 = phase.copy() # Bottom left corner
    phase_1 = np.roll(phase, -1, axis=0) # Bottom right corner
    phase_2 = np.roll(phase, -1, axis=1) # Upper left corner
    phase_3 = np.roll(phase, [-1, -1], axis=(0, 1)) # Upper right corner

    # Calculate int(rho²)
    rho_square = phase_0 ** 2 + 2 * phase_1 ** 2 + 2 * phase_2 ** 2 + phase_3 ** 2
    rho_square += phase_0 * phase_1 + phase_0 * phase_2 + 2 * phase_1 * phase_2 +\
                           phase_3 * phase_1 + phase_3 * phase_2
    rho_square = 1/12 * np.sum(rho_square) * det

    # Calculate int(rho³)
    rho_qube = phase_0 ** 3 + 2 * phase_1 ** 3 + 2 * phase_2 ** 3 + phase_3 ** 3
    rho_qube += phase_0 ** 2 * phase_1 + phase_0 ** 2 * phase_2 + phase_1 ** 2 * phase_0
    rho_qube += 2 * phase_1 ** 2 * phase_2 + phase_2 ** 2 * phase_0 + 2 * phase_2 ** 2 * phase_1
    rho_qube += phase_3 ** 2 * phase_1 + phase_3 ** 2 * phase_2 + phase_1 ** 2 * phase_3
    rho_qube += phase_2 ** 2 * phase_3
    rho_qube += phase_0 * phase_1 * phase_2 + phase_3 * phase_1 * phase_2
    rho_qube = 1/20 * np.sum(rho_qube) * det

    # Calculate int(rho⁴)
    rho_quartic = phase_0 ** 4 + 2 * phase_1 ** 4 + 2 * phase_2 ** 4 + phase_3 ** 4
    rho_quartic += phase_0 ** 3 * phase_1 + phase_0 ** 3 * phase_2 + phase_1 ** 3 * phase_0
    rho_quartic += 2 * phase_1 ** 3 * phase_2 + phase_2 ** 3 * phase_0 + 2 * phase_2 ** 3 * phase_1
    rho_quartic += phase_3 ** 3 * phase_1 + phase_3 **3 * phase_2 + phase_1 ** 3 * phase_3
    rho_quartic += phase_2 ** 3 * phase_3
    rho_quartic += phase_0 ** 2 * phase_1 ** 2 + phase_0 ** 2 * phase_2 ** 2
    rho_quartic += 2 * phase_1 ** 2 * phase_2 ** 2 + phase_3 ** 2 * phase_1 ** 2
    rho_quartic += phase_3 ** 2 * phase_2 ** 2
    rho_quartic += phase_0 ** 2 * phase_1 * phase_2 + phase_1 ** 2 * phase_0 * phase_2
    rho_quartic += phase_2 ** 2 * phase_0 * phase_1 + phase_3 ** 2 * phase_1 * phase_2
    rho_quartic += phase_1 ** 2 * phase_3 * phase_2 + phase_2 ** 2 * phase_3 * phase_1
    rho_quartic = 1/30 * np.sum(rho_quartic) * det

    return rho_square - 2 * rho_qube + rho_quartic

def phase_field_energy_sequential(phase, cell, eta):
    """ Function to calculate the phase-field energy
        on a regular 2D grid. As discretization rectangular
        triangles are used. The phase field is defined
        on the nodes.

    Parameters
    ----------
    phase: np.ndarray(nb_grid_pts) of floats
           Phase field function.
    cell: object
          muSpectre cell object
    eta: float
         Weighting parameter of the phase field energy. A larger
         eta means a broader interface.

    Returns
    -------
    energy: float
            Energy of the phase-field function
    """
    energy = eta * phase_field_gradient_sequential(phase, cell)
    energy += 1 / eta * phase_field_double_well_potential_sequential(phase, cell)
    return energy

def phase_field_energy_deriv_sequential(phase, cell, eta):
    """ Function to calculate the derivative of the
        phase-field energy on a regular 2D grid. As discretization
        rectangular triangles are used. The phase field is defined
        on the nodes.

    Parameters
    ----------
    phase: np.ndarray(nb_grid_pts) of floats
           Phase field function.
    cell: object
          muSpectre cell object
    eta: float
         Weighting parameter of the phase field energy. A larger
         eta means a broader interface.

    Returns
    -------
    derivative: np.ndarray(nb_grid_pts) of floats
                Derivative of phase_field_energy with respect to the
                phase field function at each grid point.
    """
    Lx, Ly = cell.domain_lengths
    hx = Lx / cell.nb_domain_grid_pts[0]
    hy = Ly / cell.nb_domain_grid_pts[1]

    # Derivative of the gradient term
    deriv_grad = (-np.roll(phase, -1, axis=0) + 2 * phase -\
                  np.roll(phase, 1, axis=0)) / hx ** 2
    deriv_grad += (-np.roll(phase, -1, axis=1) + 2 * phase -\
                   np.roll(phase, 1, axis=1)) / hy ** 2
    deriv_grad = 4 * deriv_grad * Lx * Ly / cell.nb_quad_pts / cell.nb_pixels

    # Derivative of the double-well potential
    det = hx * hy
    phase_up = np.roll(phase, -1, axis=1)
    phase_down = np.roll(phase, 1, axis=1)
    phase_left = np.roll(phase, 1, axis=0)
    phase_right = np.roll(phase, -1, axis=0)
    phase_up_left = np.roll(phase, [1, -1], axis=(0, 1))
    phase_down_right = np.roll(phase, [-1, 1], axis=(0, 1))

    deriv_pot = 6 * phase + phase_up + phase_down + phase_left +\
        phase_right + phase_up_left + phase_down_right
    deriv_pot = 1/6 * deriv_pot

    helper = 18 * phase ** 2 + 2 * phase_right ** 2 + 2 * phase_left ** 2
    helper += 2 * phase_up ** 2 + 2 * phase_down ** 2 + 2 * phase_up_left ** 2
    helper += 2 * phase_down_right ** 2
    helper += 4 * phase * (phase_up + phase_down + phase_right + phase_left +\
                           phase_up_left + phase_down_right)
    helper += phase_up * phase_right + phase_up * phase_up_left
    helper += phase_left * phase_up_left + phase_left * phase_down
    helper += phase_down * phase_down_right + phase_right * phase_down_right
    deriv_pot += -1/10 * helper

    helper = 24 * phase ** 3 + 2 * (phase_right ** 3 + phase_left ** 3 +\
                                    phase_up ** 3 + phase_down ** 3 +\
                                    phase_up_left ** 3 + phase_down_right ** 3)
    helper += 6 * phase ** 2 * (phase_right + phase_left + phase_up +\
                                phase_down + phase_up_left + phase_down_right)
    helper += 4 * phase * (phase_right ** 2 + phase_left ** 2 + phase_down ** 2 +\
                           phase_up ** 2 + phase_up_left ** 2 + phase_down_right ** 2)
    helper += 2 * phase * (phase_right * phase_up + phase_left * phase_up_left +\
                           phase_down * phase_down_right + phase_up * phase_up_left +\
                           phase_down * phase_left + phase_right * phase_down_right)
    helper += phase_right ** 2 * (phase_up + phase_down_right)
    helper += phase_left ** 2 * (phase_up_left + phase_down)
    helper += phase_down ** 2 * (phase_left + phase_down_right)
    helper += phase_down_right ** 2 * (phase_down + phase_right)
    helper += phase_up ** 2 * (phase_right + phase_up_left)
    helper += phase_up_left ** 2 * (phase_up + phase_left)
    deriv_pot += 1/30 * helper

    deriv_pot *= det

    # Complete derivative
    deriv = eta * deriv_grad + 1 / eta * deriv_pot
    return deriv

################################################################################
### --------------- Phase on quad pts: rectangular triangles --------------- ###
################################################################################
def gradient_rectangular_grid(phase, cell):
    """ Function to calculate the gradient part of the
        phase-field function on a regular 2D grid.
        As discretization rectangular triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_quad_pts, *nb_grid_pts) of floats
           Phase field function.
    cell: object
          muSpectre cell object

    Returns
    -------
    phase_field_gradient: float
                          Gradient term of the phase-field energy
    """
    # Lengths of subdomain
    hx = cell.domain_lengths[0] / cell.nb_domain_grid_pts[0]
    hy = cell.domain_lengths[1] / cell.nb_domain_grid_pts[1]
    Lx = hx * cell.nb_subdomain_grid_pts[0]
    Ly = hy * cell.nb_subdomain_grid_pts[1]

    # Calculate gradient
    phase = phase.reshape([cell.nb_quad_pts, *cell.nb_subdomain_grid_pts], order='F')
    helper_phase = np.average(phase, axis=0)
    dx = µ.DiscreteDerivative([0, 0], [[-1], [1]])
    dy = dx.rollaxes()
    fft = cell.fft_engine
    q = fft.fftfreq
    fourier_phase = fft.fetch_or_register_fourier_space_field(
        "fft_workspace", 1)
    fft.fft(helper_phase, fourier_phase)
    d = dx.fourier(q)
    phase_dx = np.zeros_like(helper_phase, order='f')
    fft.ifft(d * fourier_phase, phase_dx)
    phase_dx *= fft.normalisation / hx
    d = dy.fourier(q)
    phase_dy = np.zeros_like(helper_phase, order='f')
    fft.ifft(d * fourier_phase, phase_dy)
    phase_dy *= fft.normalisation / hy

    # Calculate integral of norm
    norm_gradient = phase_dx**2 + phase_dy**2
    integral_gradient = np.average(norm_gradient) * Lx * Ly

    return integral_gradient

def double_well_potential_quad_pts(phase, cell):
    """ Function to calculate the double-well-potential part
        of the phase-field function on a 2D grid.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    cell: object
          muSpectre cell object

    Returns
    -------
    phase_field_double_well: float
                             Double-well potential of the phase-field-function
    """
    # Lengths of subdomain
    hx = cell.domain_lengths[0] / cell.nb_domain_grid_pts[0]
    hy = cell.domain_lengths[1] / cell.nb_domain_grid_pts[1]
    Lx = hx * cell.nb_subdomain_grid_pts[0]
    Ly = hy * cell.nb_subdomain_grid_pts[1]

     # Double well potential
    potential = phase**2 * (1-phase)**2
    integral_potential = np.average(potential) * Lx * Ly

    return integral_potential

def phase_field_rectangular_grid(phase, eta, cell):
    """ Function to calculate the energy function of the phase-field on a 2D grid.
        As discretization rectangular triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns
    -------
    phase_field: float
                 Energy of the phase field
    """
    # Global value of face field
    phase_field = eta * gradient_rectangular_grid(phase, cell)
    phase_field += 1/eta * double_well_potential_quad_pts(phase, cell)
    phase_field = Reduction(MPI.COMM_WORLD).sum(phase_field)

    return phase_field

def phase_field_rectangular_grid_deriv_phase(phase, eta, cell):
    """ Function to calculate the partial derivative of the energy of the phase-field with respect to the phase.
        As discretization rectangular triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns
    -------
    derivative: np.ndarray(nb_pixels) of floats
                Derivative of phase_field_rectangular_grid with respect to the phase field function at each grid point.
                The iteration order of the pixels is column-major.
    """
    # Helper definitions
    dim = cell.nb_domain_grid_pts.dim
    nb_grid_pts = cell.nb_domain_grid_pts
    nb_pixels = np.prod(nb_grid_pts)
    lengths = cell.domain_lengths
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]

    # Derivative of the double well potential
    derivative = 1/eta * 2 * phase * (1-phase) * (1-2*phase) *\
                  lengths[1] * lengths[0] / nb_pixels / cell.nb_quad_pts
    derivative = derivative.reshape([cell.nb_quad_pts,
                                     *cell.nb_subdomain_grid_pts], order='F')

    # Derivative of gradient term
    phase = phase.reshape([cell.nb_quad_pts,
                           *cell.nb_subdomain_grid_pts], order='F')
    helper_phase = np.average(phase, axis=0)
    hx = lengths[0] / nb_grid_pts[0]
    hy =lengths[1] / nb_grid_pts[1]
    dx = µ.DiscreteDerivative([-1, 0], [[-1], [2], [-1]])
    dy = dx.rollaxes()
    fft = cell.fft_engine
    q = fft.fftfreq
    fourier_phase = fft.fetch_or_register_fourier_space_field(
        "fft_workspace", 1)
    fft.fft(helper_phase, fourier_phase)
    d = dx.fourier(q)
    deriv_phase_dx = np.zeros_like(helper_phase, order='f')
    fft.ifft(d * fourier_phase, deriv_phase_dx)
    deriv_phase_dx *= fft.normalisation / hx**2
    d = dy.fourier(q)
    deriv_phase_dy = np.zeros_like(helper_phase, order='f')
    fft.ifft(d * fourier_phase, deriv_phase_dy)
    deriv_phase_dy *= fft.normalisation / hy**2
    deriv_phase_dx = deriv_phase_dx.flatten(order='F')
    deriv_phase_dy = deriv_phase_dy.flatten(order='F')

    deriv_gradient = 2 * eta * (deriv_phase_dx + deriv_phase_dy) *\
        lengths[1] * lengths[0] / nb_pixels
    deriv_gradient = deriv_gradient.reshape([*cell.nb_subdomain_grid_pts], order='F')

    # Complete derivative
    deriv_gradient = deriv_gradient / cell.nb_quad_pts # Factor because I average over the quadrature points
    for i in range(cell.nb_quad_pts):
        derivative[i] += deriv_gradient

    return derivative

################################################################################
### --------------- Phase on quad pts: equilateral triangles --------------- ###
################################################################################
def gradient_equilat_tri_grid(phase, cell):
    """ Function to calculate the gradient part of the
        phase-field function on a 2D grid.
        As discretization equilateral triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    cell: object
          muSpectre cell object

    Returns
    -------
    phase_field_gradient: float
                          Gradient term of the phase-field energy
    """
    # Lengths of subdomain
    hx = cell.domain_lengths[0] / cell.nb_domain_grid_pts[0]
    hy = cell.domain_lengths[1] / cell.nb_domain_grid_pts[1]
    Lx = hx * cell.nb_subdomain_grid_pts[0]
    Ly = hy * cell.nb_subdomain_grid_pts[1]

    # Calculate gradient
    phase = phase.reshape([*cell.nb_subdomain_grid_pts], order='F')
    dx = µ.DiscreteDerivative([0, 0], [[-1], [1]])
    dy = µ.DiscreteDerivative([0, 0], [[-1/2, 1], [-1/2, 0]])
    fft = cell.fft_engine
    q = fft.fftfreq
    fourier_phase = fft.fetch_or_register_fourier_space_field(
        "fft_workspace", 1)
    fft.fft(phase, fourier_phase)
    d = dx.fourier(q)
    phase_dx = np.zeros_like(phase, order='f')
    fft.ifft(d * fourier_phase, phase_dx)
    phase_dx *= fft.normalisation / hx
    d = dy.fourier(q)
    phase_dy = np.zeros_like(phase, order='f')
    fft.ifft(d * fourier_phase, phase_dy)
    phase_dy *= fft.normalisation / hy

    # Calculate integral of norm
    norm_gradient = phase_dx**2 + phase_dy**2
    integral_gradient = np.average(norm_gradient) * Lx * Ly

    # Nondimensionalize
    #integral_gradient = integral_gradient / cell.domain_lengths[0]

    return integral_gradient

def phase_field_equilat_tri_grid(phase, eta, cell):
    """ Function to calculate the energy function of the phase-field on a 2D grid.
        As discretization equilateral triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns
    -------
    phase_field: float
                 Energy of the phase field
    """
    # Global value of face field
    phase_field = eta * gradient_equilat_tri_grid(phase, cell)
    phase_field += 1/eta * double_well_potential_quad_pts(phase, cell)
    phase_field = Reduction(MPI.COMM_WORLD).sum(phase_field)

    return phase_field

def phase_field_equilat_tri_grid_deriv_phase(phase, eta, cell):
    """ Function to calculate the partial derivative of the energy of the phase-field with respect to the phase.
        As discretization equilateral triangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns
    -------
    derivative: np.ndarray(nb_pixels) of floats
                Derivative of phase_field_equilat_tri_grid with respect to the phase field function at each grid point.
                The iteration order of the pixels is column-major.
    """
    # Helper definitions
    dim = cell.nb_domain_grid_pts.dim
    nb_grid_pts = cell.nb_domain_grid_pts
    nb_pixels = np.prod(nb_grid_pts)
    lengths = cell.domain_lengths
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]

    # Derivative of the double well potential
    derivative = 1/eta * 2 * phase * (1-phase) * (1-2*phase) *\
                     lengths[1] * lengths[0] / nb_pixels

    # Derivative of gradient term
    phase = phase.reshape([*cell.nb_subdomain_grid_pts], order='F')
    hx = lengths[0] / nb_grid_pts[0]
    hy =lengths[1] / nb_grid_pts[1]
    dx = µ.DiscreteDerivative([-1, 0], [[-2], [4], [-2]])
    dy = µ.DiscreteDerivative([-1, -1], [[0, 1/2, -1], [-1, 3, -1], [-1, 1/2, 0]])
    fft = cell.fft_engine
    q = fft.fftfreq
    fourier_phase = fft.fetch_or_register_fourier_space_field(
        "fft_workspace", 1)
    fft.fft(phase, fourier_phase)
    d = dx.fourier(q)
    deriv_phase_dx = np.zeros_like(phase, order='f')
    fft.ifft(d * fourier_phase, deriv_phase_dx)
    deriv_phase_dx *= fft.normalisation / hx**2
    d = dy.fourier(q)
    deriv_phase_dy = np.zeros_like(phase, order='f')
    fft.ifft(d * fourier_phase, deriv_phase_dy)
    deriv_phase_dy *= fft.normalisation / hy**2
    deriv_phase_dx = deriv_phase_dx.flatten(order='F')
    deriv_phase_dy = deriv_phase_dy.flatten(order='F')

    derivative += eta * (deriv_phase_dx + deriv_phase_dy) *\
                  lengths[1] * lengths[0] / nb_pixels

    return derivative
