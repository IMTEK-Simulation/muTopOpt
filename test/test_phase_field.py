#
# Copyright 2021 Lars Pastewka
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
Tests the phase field functions
"""
import numpy as np
import muSpectre as µ

from muTopOpt.PhaseField import phase_field_rectangular_grid_deriv_phase
from muTopOpt.PhaseField import phase_field_rectangular_grid
from muTopOpt.PhaseField import phase_field_equilat_tri_grid_deriv_phase
from muTopOpt.PhaseField import phase_field_equilat_tri_grid
from muTopOpt.PhaseField import phase_field_energy_sequential
from muTopOpt.PhaseField import phase_field_double_well_potential_sequential
from muTopOpt.PhaseField import phase_field_energy_deriv_sequential

################################################################################
### ---------------------------- Phase on nodes ---------------------------- ###
################################################################################
def test_double_well_potential():
    # Set up
    nb_grid_pts = [5, 7]
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    phase = np.random.random(nb_grid_pts)

    # Direct calculation of double_well_potential
    Jacobian_matrix = np.diag(np.asarray(lengths) / np.asarray(nb_grid_pts))
    Jacobian_det = np.linalg.det( Jacobian_matrix)

    rho_squared_pixel = lambda rho0, rho1, rho2, rho3: (1 / 12) * (rho0 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho0 * rho1 + rho0 * rho2 + rho2 * rho1) \
                                                       + (1 / 12) * (rho3 ** 2 + rho1 ** 2 + rho2 ** 2) \
                                                       + (2 / 24) * (rho3 * rho1 + rho3 * rho2 + rho2 * rho1)

    rho_squared = np.sum(rho_squared_pixel(phase,
                                           np.roll(phase, -1, axis=(0)),
                                           np.roll(phase, -1, axis=(1)),
                                           np.roll(phase, -1 * np.array([1, 1]),
                                                   axis=(0, 1)))) * Jacobian_det

    rho_qubed_pixel = lambda rho0, rho1, rho2, rho3: (1 / 20) * (rho0 ** 3 + rho1 ** 3 + rho2 ** 3) \
                                                     + (1 / 20) * (rho3 ** 3 + rho1 ** 3 + rho2 ** 3) \
                                                     + (3 / 60) * (rho0 ** 2 * rho1 + rho0 ** 2 * rho2 \
                                                                   + rho1 ** 2 * rho0 + rho1 ** 2 * rho2 \
                                                                   + rho2 ** 2 * rho0 + rho2 ** 2 * rho1) \
                                                     + (3 / 60) * (rho3 ** 2 * rho1 + rho3 ** 2 * rho2 \
                                                                  + rho1 ** 2 * rho3 + rho1 ** 2 * rho2 \
                                                                   + rho2 ** 2 * rho3 + rho2 ** 2 * rho1) \
                                                    + (6 / 120) * (rho0 * rho1 * rho2) \
                                                    + (6 / 120) * (rho3 * rho1 * rho2)
    rho_qubed = np.sum(rho_qubed_pixel(phase,
                                       np.roll(phase, -1, axis=(0)),
                                       np.roll(phase, -1, axis=(1)),
                                       np.roll(phase, -1 * np.array([1, 1]),
                                               axis=(0, 1)))) * Jacobian_det

    rho_quartic_pixel = lambda rho0, rho1, rho2, rho3: (1 / 30) * (rho0 ** 4 + rho1 ** 4 + rho2 ** 4) \
                                                       + (4 / 120) * (rho0 ** 3 * rho1 + rho0 ** 3 * rho2 \
                                                                      + rho1 ** 3 * rho0 + rho1 ** 3 * rho2 \
                                                                      + rho2 ** 3 * rho0 + rho2 ** 3 * rho1) \
                                                       + (6 / 180) * (rho0 ** 2 * rho1 ** 2 \
                                                                      + rho0 ** 2 * rho2 ** 2 \
                                                                      + rho1 ** 2 * rho2 ** 2) \
                                                       + (12 / 360) * (rho0 ** 2 * rho1 * rho2 \
                                                                       + rho0 * rho1 ** 2 * rho2 \
                                                                       + rho0 * rho1 * rho2 ** 2) \
                                                       + (1 / 30) * (rho3 ** 4 + rho1 ** 4 + rho2 ** 4) \
                                                       + (4 / 120) * (rho3 ** 3 * rho1 + rho3 ** 3 * rho2 \
                                                                      + rho1 ** 3 * rho3 + rho1 ** 3 * rho2 \
                                                                      + rho2 ** 3 * rho3 + rho2 ** 3 * rho1) \
                                                       + (6 / 180) * (rho3 ** 2 * rho1 ** 2 \
                                                                      + rho3 ** 2 * rho2 ** 2 \
                                                                      + rho1 ** 2 * rho2 ** 2) \
                                                       + (12 / 360) * (rho3 ** 2 * rho1 * rho2 \
                                                                       + rho3 * rho1 ** 2 * rho2 \
                                                                       + rho3 * rho1 * rho2 ** 2)
    rho_quartic = np.sum(rho_quartic_pixel(phase,
                                           np.roll(phase, -1, axis=(0)),
                                           np.roll(phase, -1, axis=(1)),
                                           np.roll(phase, -1 * np.array([1, 1]),
                                                   axis=(0, 1)))) * Jacobian_det
    # (ρ^2 (1 - ρ)^2) = ρ^2 - 2ρ^3 + ρ^4
    integral_1 = rho_squared - 2 * rho_qubed + rho_quartic

    # Calculation with function call
    integral_2 = phase_field_double_well_potential_sequential(phase, cell)

    #print(f'integral_1 = {integral_1}')
    #print(f'integral_2 = {integral_2}')
    assert abs(integral_1 - integral_2) < 1e-6


def test_phase_field_energy_deriv(plot=False):
    # Set up
    nb_grid_pts = [5, 7]
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    eta = 0.8

    phase = np.random.random(nb_grid_pts)

    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    # Analytical calculation of the derivative
    func = phase_field_energy_sequential(phase, cell, eta)
    deriv = phase_field_energy_deriv_sequential(phase, cell, eta)

    # Finite difference calculation of the derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                phase[i, j] += delta
                func_plus = phase_field_energy_sequential(phase, cell, eta)
                deriv_fin_diff[i, j] = (func_plus - func) / delta
                phase[i, j] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    #print('Derivs:')
    #print(deriv[1, 1])
    #print(deriv_fin_diff[1, 1])

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.suptitle('Phase field defined on nodes')
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of phase field derivative')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6


################################################################################
### --------------------------- Phase on quad pts -------------------------- ###
################################################################################
def test_phase_field_rectangular_grid_deriv_phase(plot=False):
    """ Check the implementation of the derivative of the phase field
        term with respect to the phase on a rectangular grid.
    """
    # Set up
    nb_grid_pts = [5, 7]
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    eta = 0.8

    phase = np.random.random([nb_quad_pts, *nb_grid_pts]).flatten(order='F')

    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    # Calculate derivative
    ph_field = phase_field_rectangular_grid(phase, eta, cell)
    deriv = phase_field_rectangular_grid_deriv_phase(phase, eta, cell)
    deriv = deriv.flatten(order='F')

    # Finite difference calculation of the derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(deriv)):
            phase[i] += delta
            ph_field_plus = phase_field_rectangular_grid(phase, eta, cell)
            deriv_fin_diff[i] = (ph_field_plus - ph_field) / delta
            phase[i] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of phase field derivative')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6


def test_phase_field_equilat_tri_grid_deriv_phase(plot=False):
    """ Check the implementation of the derivative of the phase field
        term with respect to the phase on a grid of equilateral triangles.
    """
    # Set up
    nb_grid_pts = [5, 7]
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    eta = 0.8

    phase = np.random.random(nb_grid_pts).flatten(order='F')

    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    # Calculate derivative
    ph_field = phase_field_equilat_tri_grid(phase, eta, cell)
    deriv = phase_field_equilat_tri_grid_deriv_phase(phase, eta, cell)

    # Finite difference calculation of the derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(deriv)):
            phase[i] += delta
            ph_field_plus = phase_field_equilat_tri_grid(phase, eta, cell)
            deriv_fin_diff[i] = (ph_field_plus - ph_field) / delta
            phase[i] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of phase field derivative')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

if __name__ == "__main__":
    test_double_well_potential()
    test_phase_field_energy_deriv(plot=True)
    test_phase_field_rectangular_grid_deriv_phase(plot=True)
    test_phase_field_equilat_tri_grid_deriv_phase(plot=True)

