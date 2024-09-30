"""
Tests the wrapper function.
"""
import sys
import os

import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ
import muFFT

from muTopOpt.Controller import wrapper
from muTopOpt.AimFunction import aim_function
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential

def test_wrapper(plot=False):
    """ Test the function and the sensitivity
        calculated with the wrapper function.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [3, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_pixels = np.prod(nb_grid_pts)

    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    nb_quad_pts = cell.nb_quad_pts

    # Material
    np.random.seed(1)
    phase = np.random.random(nb_grid_pts)
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2.4

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.02
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.1

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (tol, maxiter, verbose)
    solver_args = (tol, tol, verbose)

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    ### ----- Target stresses ----- ###
    # Stresses for homogenous material with average parameters
    lame1_av = delta_lame1 / 2
    lame2_av = delta_lame2 / 2
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * lame2_av * DelF + lame1_av * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Correct aim function? ----- ###
    # Material initialization
    density = node_to_quad_pt_2_quad_pts_sequential(phase)
    density = density.reshape([nb_quad_pts, nb_pixels], order='F')
    lame1 = delta_lame1 * density ** order
    lame2 = delta_lame2 * density ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    krylov_solver = µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, krylov_solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    # Aim function
    aim2 = aim_function(cell, phase, strains, stresses, *aim_args)

    # density = density.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    aim, S = wrapper(phase, cell, mat, delta_lame1, delta_lame2, 0, 0, order,
                     DelFs, krylov_solver_args, solver_args, aim_args,
                     calc_sensitivity=True, folder=None)
    assert abs(aim - aim2) <= 1e-7

    ### ---- Test sensitivity ----- ###
    # Finite difference calculation of the sensitivity
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    S = S.flatten(order='F')
    phase = phase.flatten(order='F')
    for delta in delta_list:
        deriv_fin_diff = np.empty(S.shape)
        # Iterate over quadrature points
        for pixel_id, pixel in cell.pixels.enumerate():
            # Disturb material
            phase[pixel_id] += delta
            # New aim function
            aim_plus = wrapper(phase, cell, mat, delta_lame1, delta_lame2,
                               0, 0, order, DelFs, krylov_solver_args,
                               solver_args, aim_args,
                               calc_sensitivity=False, folder=None)
            deriv_fin_diff[pixel_id] = (aim_plus - aim) / delta
            phase[pixel_id] -= delta

        diff = np.linalg.norm(deriv_fin_diff - S)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plotting (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        fig.suptitle('Test wrapper')
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_wrapper_per_quad_pts(plot=False):
    """ Test the function and the sensitivity
        calculated with the wrapper function.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [3, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_pixels = np.prod(nb_grid_pts)

    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    nb_quad_pts = cell.nb_quad_pts

    # Material
    np.random.seed(1)
    phase = np.random.random([nb_quad_pts, nb_pixels])
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2.4

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.02
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.1

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (tol, maxiter, verbose)
    solver_args = (tol, tol, verbose)

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    ### ----- Target stresses ----- ###
    # Stresses for homogenous material with average parameters
    lame1_av = delta_lame1 / 2
    lame2_av = delta_lame2 / 2
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * lame2_av * DelF + lame1_av * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Correct aim function? ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    krylov_solver = µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, krylov_solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    # Aim function
    aim2 = aim_function(cell, phase, strains, stresses, *aim_args)

    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    aim, S = wrapper(phase, cell, mat, delta_lame1, delta_lame2, 0, 0, order,
                     DelFs, krylov_solver_args, solver_args, aim_args,
                     calc_sensitivity=True, folder=None)
    assert abs(aim - aim2) <= 1e-7

    ### ---- Test sensitivity ----- ###
    # Finite difference calculation of the sensitivity
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(S.shape)
        # Iterate over quadrature points
        for pixel_id, pixel in cell.pixels.enumerate():
            for quad_id in range(cell.nb_quad_pts):
                # Disturb material
                index = (*tuple(pixel), quad_id)
                quad_index = cell.nb_quad_pts * pixel_id + quad_id
                phase[(quad_id, *tuple(pixel))] += delta

                # New aim function
                aim_plus = wrapper(phase, cell, mat, delta_lame1, delta_lame2,
                                   0, 0, order, DelFs, krylov_solver_args,
                                   solver_args, aim_args,
                                   calc_sensitivity=False, folder=None)
                deriv_fin_diff[quad_index] = (aim_plus - aim) / delta
                phase[(quad_id, *tuple(pixel))] -= delta

        diff = np.linalg.norm(deriv_fin_diff - S)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plotting (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

if __name__ == "__main__":
    test_wrapper(plot=True)
