"""
This file contains the phase-field aim function and gradients.
"""
import numpy as np
import muSpectre as µ

from NuMPI import MPI
from NuMPI.Tools import Reduction

def phase_field_rectangular_grid(phase, eta, cell):
    """ Function to calculate the energy of the phase-field on a 2D grid.
        As discretization rectangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns:
    phase_field: float
                 Energy of the phase field
    """
    # Lengths of subdomain
    hx = cell.domain_lengths[0] / cell.nb_domain_grid_pts[0]
    hy = cell.domain_lengths[1] / cell.nb_domain_grid_pts[1]
    Lx = hx * cell.nb_subdomain_grid_pts[0]
    Ly = hy * cell.nb_subdomain_grid_pts[1]

    # Double well potential
    potential = phase**2 * (1-phase)**2
    integral_potential = np.average(potential) * Lx * Ly

    # Gradient part
    phase = phase.reshape([*cell.nb_subdomain_grid_pts], order='F')
    dx = µ.DiscreteDerivative([0, 0], [[-1], [1]])
    dy = dx.rollaxes()
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

    # Together
    norm_gradient = phase_dx**2 + phase_dy**2
    integral_gradient = np.average(norm_gradient) * Lx * Ly

    # Global value of face field
    phase_field = eta * integral_gradient + 1/eta * integral_potential
    phase_field = Reduction(MPI.COMM_WORLD).sum(phase_field)

    return phase_field

def phase_field_rectangular_grid_deriv_phase(phase, eta, cell):
    """ Function to calculate the partial derivative of the energy of the phase-field with respect to the phase.
        As discretization 2D-rectangles are used.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    eta: float
         Weighting parameter of the phase field energy. A larger eta means a broader interface.
    cell: object
          muSpectre cell object

    Returns:
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
                  lengths[0] * lengths[1] / nb_pixels

    # Derivative of gradient term
    phase = phase.reshape([*cell.nb_subdomain_grid_pts], order='F')
    hx = lengths[0] / nb_grid_pts[0]
    hy =lengths[1] / nb_grid_pts[1]
    dx = µ.DiscreteDerivative([-1, 0], [[-1], [2], [-1]])
    dy = dx.rollaxes()
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

    derivative += 2 * eta * (deriv_phase_dx + deriv_phase_dy) *\
                  lengths[0] * lengths[1] / nb_pixels

    return derivative
