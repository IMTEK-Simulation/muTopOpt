"""
Tests the partial derivatives of the aim function
"""
import sys
import os

import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ

import muTopOpt.AimFunction as aim_func
from muTopOpt.Controller import calculate_dstress_dmat
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative_sequential

def test_aim_deriv_strain(plot=False):
    """ Check the the derivative of the aim function
        with respect to the strains for one load case and one
        quadrature point.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    np.random.seed(1)
    phase = np.random.random(nb_grid_pts)
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.02

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.08

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

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

    ### ----- Analytical derivative ----- ###
    # Material initialization
    density = phase.flatten(order='F')
    lame1 = delta_lame1 * density ** order
    lame2 = delta_lame2 * density ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])
    cell.initialise()

    # muSpectre calculation
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    helper_stresses = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        helper_stresses.append(stress.copy())
        strain = r.grad.copy()
        strains.append(strain)

    # Function and derivative
    aim = aim_func.aim_function(cell, phase, strains, stresses,
                                target_stresses, eta, weight_phase_field)
    derivs = aim_func.aim_function_deriv_strain(cell, strains, stresses,
                                                target_stresses, eta,
                                                weight_phase_field)

    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case]
            deriv_fin_diff = np.empty((deriv.shape))
            for i in range(len(deriv)):
                strain[i] += delta
                helper_stresses[i_case] =\
                    cell.evaluate_stress(strain.reshape(shape, order='F'))
                aim_plus =\
                    aim_func.aim_function(cell, phase, strains,
                                          helper_stresses,
                                          target_stresses, eta,
                                          weight_phase_field)
                deriv_fin_diff[i] = (aim_plus - aim) / delta
                strain[i] -= delta
            helper_stresses[i_case] = stresses[i_case].copy()
            diff += (deriv_fin_diff - deriv)
        diff = np.linalg.norm(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of partial deriv StressTarget w.r.t. strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6# TODO

def test_aim_deriv_strain_two(plot=False):
    """ Check the the derivative of the aim function
        with respect to the strains for two load cases
        and two quadrature points.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    # Material
    phase = np.random.random(nb_grid_pts)
    lambda_1 = 0.4
    lambda_0 = 0.1
    mu_1 = 120
    mu_0 = 10
    order = 2.1

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.09

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.013
    DelFs[0][0, 0] = loading
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    ### ----- Target stresses ----- ###
    # Stresses for homogenous material with average parameters
    lame1_av = (lambda_1 - lambda_0) / 2
    lame2_av = (mu_1 - mu_0) / 2
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * lame2_av * DelF + lame1_av * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    density = node_to_quad_pt_2_quad_pts_sequential(phase)
    density = density.reshape([nb_quad_pts, nb_pixels], order='F')
    lame1 = (lambda_1 - lambda_0) * density ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density ** order + mu_0
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    helper_stresses = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        helper_stresses.append(stress.copy())
        strain = r.grad.copy()
        strains.append(strain)

    # Function and derivative
    aim = aim_func.aim_function(cell, phase, strains, stresses,
                                target_stresses, eta, weight_phase_field)
    derivs = aim_func.aim_function_deriv_strain(cell, strains, stresses,
                                                target_stresses, eta,
                                                weight_phase_field)

    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case]
            deriv_fin_diff = np.empty((deriv.shape))
            for i in range(len(deriv)):
                strain[i] += delta
                helper_stresses[i_case] =\
                    cell.evaluate_stress(strain.reshape(shape, order='F'))
                aim_plus =\
                    aim_func.aim_function(cell, phase, strains,
                                          helper_stresses, target_stresses,
                                          eta, weight_phase_field)
                deriv_fin_diff[i] = (aim_plus - aim) / delta
                strain[i] -= delta
            helper_stresses[i_case] = stresses[i_case].copy()
            diff += (deriv_fin_diff - deriv)
        diff = np.linalg.norm(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of partial deriv AimFunction w.r.t. strains (two strains)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_aim_deriv_phase(plot=False):
    """ Check the the derivative of the aim function
        with respect to the phase for one load case and one
        quadrature point.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [3, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    nb_pixels = np.prod(nb_grid_pts)
    cell = µ.Cell(nb_grid_pts, lengths, formulation)
    nb_quad_pts = cell.nb_quad_pts

    # Material
    np.random.seed(1)
    phase = np.random.random([nb_quad_pts, nb_pixels])
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.01

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.011

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

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

    ### ----- Analytical derivative ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id],
                           lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    helper_stresses = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, phase,
                                               delta_lame1, delta_lame2,
                                               0, 0, order=order)
    aim = aim_func.aim_function(cell, phase, strains, stresses,
                                target_stresses, eta, weight_phase_field)
    deriv = aim_func.aim_function_deriv_phase(cell, phase, strains, stresses,
                                              dstress_dmat_list,
                                              target_stresses, eta,
                                              weight_phase_field)

    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)

        # Iterate over quadrature points
        for pixel_id, pixel in cell.pixels.enumerate():
            for quad_id in range(cell.nb_quad_pts):
                # Disturb material
                index = (*tuple(pixel), quad_id)
                quad_index = cell.nb_quad_pts * pixel_id + quad_id
                lame1_new = (phase[quad_index] + delta) ** order * delta_lame1
                lame2_new = (phase[quad_index] + delta) ** order * delta_lame2
                mat.set_lame_constants(quad_index, lame1_new, lame2_new)
                phase[quad_index] += delta

                # New stresses
                stresses = []
                for strain in strains:
                    stress = cell.evaluate_stress(strain.reshape(shape, order='F'))
                    stresses.append(stress.copy())

                # Derivative of square_error_target_stresses()
                aim_plus = aim_func.aim_function(cell, phase, strains,
                                                 stresses, target_stresses,
                                                 eta, weight_phase_field)
                deriv_fin_diff[(quad_id, *tuple(pixel))] = (aim_plus - aim) / delta
                mat.set_lame_constants(quad_index, lame1[quad_id, pixel_id],
                                       lame2[quad_id, pixel_id])
                phase[quad_index] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of partial deriv StressTarget w.r.t. phase')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_aim_deriv_phase_two(plot=False):
    """ Check the the derivative of the aim function
        with respect to the phase for two load cases and two
        quadrature points.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [3, 7]
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
    order = 2

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.1

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.011

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

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

    ### ----- Analytical derivative ----- ###
    # Material initialization
    density = node_to_quad_pt_2_quad_pts_sequential(phase)
    density = density.reshape([nb_quad_pts, nb_pixels], order='F')
    lame1 = delta_lame1 * density ** order
    lame2 = delta_lame2 * density ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id],
                           lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    helper_stresses = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    density = density.reshape([nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, density,
                                               delta_lame1, delta_lame2,
                                               0, 0, order=order)
    aim = aim_func.aim_function(cell, phase, strains, stresses,
                                target_stresses, eta, weight_phase_field)
    deriv = aim_func.aim_function_deriv_phase(cell, phase, strains, stresses,
                                              dstress_dmat_list,
                                              target_stresses, eta,
                                              weight_phase_field)


    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                # Disturb material
                phase[i, j] += delta
                density = node_to_quad_pt_2_quad_pts_sequential(phase)
                density = density.reshape([nb_quad_pts, nb_pixels], order='F')
                lame1 = delta_lame1 * density ** order
                lame2 = delta_lame2 * density ** order
                for pixel_id, pixel in cell.pixels.enumerate():
                    for q in range(cell.nb_quad_pts):
                        quad_id = cell.nb_quad_pts * pixel_id + q
                        mat.set_lame_constants(quad_id, lame1[q, pixel_id],
                                               lame2[q, pixel_id])
                # New stresses
                stresses = []
                for strain in strains:
                    stress = cell.evaluate_stress(strain.reshape(shape, order='F'))
                    stresses.append(stress.copy())
                # Derivative of aim function
                aim_plus = aim_func.aim_function(cell, phase, strains,
                                                 stresses, target_stresses,
                                                 eta, weight_phase_field)
                deriv_fin_diff[i, j] = (aim_plus - aim) / delta
                phase[i, j] -= delta
        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of partial deriv StressTarget w.r.t. phase')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

if __name__ == "__main__":
    test_aim_deriv_strain(plot=True)
    test_aim_deriv_strain_two(plot=True)
    test_aim_deriv_phase(plot=True)
    test_aim_deriv_phase_two(plot=True)
