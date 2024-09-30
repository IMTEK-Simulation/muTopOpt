"""
Tests the derivative of the stress with respect to the material density.
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

from muTopOpt.Controller import calculate_dstress_dmat

def test_dstress_dmat(plot=False):
    """ Test the calculation of dstress_dphase for a simple
        case: one quadrature point, one loading case.
    """
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [3, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_lame1 = 0.4
    delta_lame2 = 120
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    derivs = calculate_dstress_dmat(cell, mat, strains, phase,
                                      delta_lame1, delta_lame2,
                                      0, 0, order=order)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2, order='F')
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            for pixel_id in cell.pixel_indices:
                lame1_new = (phase[pixel_id] + delta) ** order * delta_lame1
                lame2_new = (phase[pixel_id] + delta) ** order * delta_lame2
                mat.set_lame_constants(pixel_id, lame1_new, lame2_new)
                # Derivative
                stress_plus = cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] =\
                    (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta
                mat.set_lame_constants(pixel_id, lame1[pixel_id], lame2[pixel_id])
            diff += np.linalg.norm(deriv - deriv_fin_diff)**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of stress deriv density')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6


def test_dstress_dmat_void(plot=False):
    """ Test the calculation of dstress_dphase with one
        pixel having 0 stiffness.
    """
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    phase[0] = 0
    delta_lame1 = 0.4
    delta_lame2 = 120
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    derivs = calculate_dstress_dmat(cell, mat, strains, phase,
                                      delta_lame1, delta_lame2,
                                      0, 0, order=order)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2, order='F')
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            for pixel_id in cell.pixel_indices:
                lame1_new = (phase[pixel_id] + delta) ** order * delta_lame1
                lame2_new = (phase[pixel_id] + delta) ** order * delta_lame2
                mat.set_lame_constants(pixel_id, lame1_new, lame2_new)
                # Derivative
                stress_plus = cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] =\
                    (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta
                mat.set_lame_constants(pixel_id, lame1[pixel_id], lame2[pixel_id])
            diff += np.linalg.norm(deriv - deriv_fin_diff)**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of stress deriv density (with void)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_dstress_dmat_two_strains(plot=False):
    """ Test the calculation of dstress_dphase with one
        quadrature point and two load cases.
    """
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    phase[0] = 0
    delta_lame1 = 0.4
    delta_lame2 = 120
    order = 3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01
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

    ### ----- Analytical derivation ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    derivs = calculate_dstress_dmat(cell, mat, strains, phase,
                                      delta_lame1, delta_lame2,
                                      0, 0, order=order)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2, order='F')
            stress = stresses[i_case].reshape(deriv.shape, order='F')
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
                    # Derivative
                    stress_plus = cell.evaluate_stress(strain.reshape(shape, order='F'))
                    stress_plus = stress_plus.reshape(deriv.shape, order='F')
                    deriv_fin_diff[:, :, quad_id, pixel_id] =\
                        (stress_plus[:, :, quad_id, pixel_id] - stress[:, :, quad_id, pixel_id]) / delta
                    mat.set_lame_constants(quad_index, lame1[quad_index], lame2[quad_index])
            diff += np.linalg.norm(deriv - deriv_fin_diff) ** 2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of stress deriv density (two strains)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_dstress_dmat_2_quad_pts(plot=False):
    """ Test the calculation of dstress_dphase with two
        quadrature points and two load cases.
    """
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    weights=[1, 1]
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    # Material
    phase = np.random.random([nb_quad_pts, nb_pixels])
    lambda_1 = 0.4
    lambda_0 = 0.1
    mu_1 = 120
    mu_0 = 10
    order = 2.1

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01
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


    ### ----- Analytical derivation ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights=weights)

    # Material initialization
    lame1 = (lambda_1 - lambda_0) * phase ** order + lambda_0
    lame2 = (mu_1 - mu_0) * phase ** order + mu_0
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id],
                           lame2[:, pixel_id])
    cell.initialise()

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    derivs = calculate_dstress_dmat(cell, mat, strains, phase,
                                      lambda_1, mu_1,
                                      lambda_0, mu_0, order=order)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2, order='F')
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            # Iterate over quadrature points
            for pixel_id, pixel in cell.pixels.enumerate():
                for quad_id in range(cell.nb_quad_pts):
                    # Disturb material
                    index = (*tuple(pixel), quad_id)
                    quad_index = cell.nb_quad_pts * pixel_id + quad_id
                    lame1_new = (phase[quad_index] + delta) ** order * (lambda_1 - lambda_0) + lambda_0
                    lame2_new = (phase[quad_index] + delta) ** order * (mu_1 - mu_0) + mu_0
                    mat.set_lame_constants(quad_index, lame1_new, lame2_new)
                    # Derivative
                    stress_plus = cell.evaluate_stress(strain.reshape(shape, order='F'))
                    stress_plus = stress_plus.reshape(deriv.shape, order='F')
                    deriv_fin_diff[:, :, quad_id, pixel_id] =\
                        (stress_plus[:, :, quad_id, pixel_id] - stress[:, :, quad_id, pixel_id]) / delta
                    mat.set_lame_constants(quad_index, lame1[quad_id, pixel_id], lame2[quad_id, pixel_id])
            diff += np.linalg.norm(deriv - deriv_fin_diff) ** 2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of stress deriv density (two quad pts)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6


if __name__ == "__main__":
    test_dstress_dmat(plot=True)
    test_dstress_dmat_void(plot=True)
    test_dstress_dmat_two_strains(plot=True)
    test_dstress_dmat_2_quad_pts(plot=True)
