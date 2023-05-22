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


def test_phase_field_rectangular_grid_deriv_phase(plot=False):
    """ Check the implementation of the derivative of the phase field
        term with respect to the phase on a rectangular grid.
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
    ph_field = phase_field_rectangular_grid(phase, eta, cell)
    deriv = phase_field_rectangular_grid_deriv_phase(phase, eta, cell)

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

    # Exponential fit
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of phase field derivative')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='o', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6
