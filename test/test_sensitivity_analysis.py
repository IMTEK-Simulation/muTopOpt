"""
Tests the calculation of the sensitivity.
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
from muTopOpt.Controller import sensitivity_analysis_per_quad_pts
from muTopOpt.Controller import sensitivity_analysis_per_grid_pts
from muTopOpt.Controller import calculate_dstress_dmat
from muTopOpt.AimFunction import aim_function
from muTopOpt.AimFunction import aim_function_deriv_strain
from muTopOpt.AimFunction import aim_function_deriv_phase
from muTopOpt.PhaseField import phase_field_rectangular_grid as calculate_phase_field
from muTopOpt.PhaseField import phase_field_rectangular_grid_deriv_phase as\
    calculate_phase_field_deriv
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative_sequential

def test_sensitivity_only_phase_field(plot=False):
    """ Test the sensitivity considering only the phase-field
        term of the aim function. This means the derivative
        with respect to the strains is 0.
    """
    ### ----- Aim function + partial derivatives ----- ###
    def aim_phase(phase, cell, eta):
        return calculate_phase_field(phase, eta, cell)

    def aim_phase_deriv_strains(phase):
        return len(strains) * [np.zeros(strains[0].shape)]

    def aim_phase_deriv_phase(phase, cell, eta):
        return calculate_phase_field_deriv(phase, eta, cell)

    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_lame1 = 0.4
    delta_lame2 = 120
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.01
    DelFs[0][0, 0] = loading
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2
    nb_strain_steps = 1

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (tol, maxiter, verbose)
    solver_args = (tol, tol, verbose)

    # Phase field weighting parameter
    eta = 0.2

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

   # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])
    cell.initialise()

    # Calculate aim function
    aim = aim_phase(phase, cell, eta)

    # Calculate partial derivatives
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    for DelF in DelFs:
        res = µ.solvers.newton_cg(cell, DelF, krylov_solver, *solver_args)
        strains.append(res.grad.copy())
        # stresses.append(res.stress.copy())
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, phase,
                                                   delta_lame1, delta_lame2, 0,
                                                   0, order=order)

    deriv_strains = aim_phase_deriv_strains(phase)
    deriv_phase = aim_phase_deriv_phase(phase, cell, eta)

    # Sensitivity analysis
    deriv = sensitivity_analysis_per_quad_pts(cell, krylov_solver, strains,
                                 deriv_strains, deriv_phase, dstress_dmat_list, tol)
    deriv = deriv.flatten(order='F')

    ### ----- Finite difference derivation ----- ###
    aim = aim_phase(phase, cell, eta)
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = aim_phase(phase, cell, eta)
            deriv_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

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
        ax.set_ylabel('Abs error of sensitivity (only phase-field)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_sensitivity_analysis_per_quad_pts(plot=False):
    """ Test the sensitivity for one load case and one
        quadrature point.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    np.random.seed(1)
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.02

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.14

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
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    lame1 = delta_lame1 * phase ** order
    lame2 = delta_lame2 * phase ** order
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[pixel_id], lame2[pixel_id])
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
    aim = aim_function(cell, phase, strains, stresses, *aim_args)

    # Partial derivatives
    phase = phase.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, phase,
                                                   delta_lame1, delta_lame2, 0,
                                                   0, order=order)

    deriv_strains = aim_function_deriv_strain(cell, strains, stresses, *aim_args)
    deriv_phase = aim_function_deriv_phase(cell, phase, strains, stresses,
                                           dstress_dmat_list, *aim_args)

    # Sensitivity analysis
    deriv = sensitivity_analysis_per_quad_pts(cell, krylov_solver, strains,
                                 deriv_strains, deriv_phase, dstress_dmat_list, tol)
    deriv = deriv.flatten(order='F')

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    phase = phase.flatten(order='F')
    lame1 = lame1.flatten(order='F')
    lame2 = lame2.flatten(order='F')
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(phase)):
            # Disturb material
            phase[i] += delta
            lame1_new = phase[i] ** order * delta_lame1
            lame2_new = phase[i] ** order * delta_lame2
            mat.set_lame_constants(i, lame1_new, lame2_new)
            # Calculate new strains + stresses
            stresses = []
            strains = []
            for DelF in DelFs:
                r = µ.solvers.newton_cg(cell, DelF, krylov_solver, tol, tol, verbose)
                stress = r.stress.copy()
                stresses.append(stress)
                strain = r.grad.copy()
                strains.append(strain)
            # New aim function
            aim_plus = aim_function(cell, phase, strains, stresses, *aim_args)
            deriv_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta
            mat.set_lame_constants(i, lame1[i], lame2[i])

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
        ax.set_ylabel('Abs error of sensitivity')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-5

def test_sensitivity_analysis_per_grid_pts(plot=False):
    """ Test the sensitivity for one load case and
        two quadrature points.
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
    order = 2

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.02

    # Phase field weighting parameter
    eta = 0.2
    weight_phase_field = 0.2

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
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Analytical derivative ----- ###
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
    aim = aim_function(cell, phase, strains, stresses, *aim_args)

    # Partial derivatives
    density = density.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, density,
                                               delta_lame1, delta_lame2, 0,
                                               0, order=order)

    deriv_strains = aim_function_deriv_strain(cell, strains, stresses, *aim_args)
    deriv_phase = aim_function_deriv_phase(cell, phase, strains, stresses,
                                           dstress_dmat_list, *aim_args)

    # Sensitivity analysis
    deriv = sensitivity_analysis_per_grid_pts(cell, krylov_solver, strains,
                                 deriv_strains, deriv_phase, dstress_dmat_list, tol)
    deriv = deriv.reshape(nb_grid_pts, order='F')

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        # Iterate over quadrature points
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                phase[i, j] += delta
                density = node_to_quad_pt_2_quad_pts_sequential(phase)
                density = density.reshape([nb_quad_pts, nb_pixels], order='F')
                lame1 = delta_lame1 * density ** order
                lame2 = delta_lame2 * density ** order
                for pixel_id, pixel2 in cell.pixels.enumerate():
                    for quad_id in range(cell.nb_quad_pts):
                        quad = pixel_id * nb_quad_pts + quad_id
                        mat.set_lame_constants(quad, lame1[quad_id, pixel_id],
                                               lame2[quad_id, pixel_id])

                # Calculate new strains + stresses
                stresses = []
                strains = []
                for DelF in DelFs:
                    r = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                            tol, tol, verbose)
                    stress = r.stress.copy()
                    stresses.append(stress)
                    strain = r.grad.copy()
                    strains.append(strain)

                # New aim function
                aim_plus = aim_function(cell, phase, strains, stresses, *aim_args)
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
        fig.suptitle('Two quadrature points')
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

def test_sensitivity_analysis_per_grid_pts_two(plot=False):
    """ Test the sensitivity for two load cases and
        two quadrature points.
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
    weight_phase_field = 0.3

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
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Analytical derivative ----- ###
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
    aim = aim_function(cell, phase, strains, stresses, *aim_args)

    # Partial derivatives
    density = density.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, density,
                                                   delta_lame1, delta_lame2, 0,
                                                   0, order=order)

    deriv_strains = aim_function_deriv_strain(cell, strains, stresses, *aim_args)
    deriv_phase = aim_function_deriv_phase(cell, phase, strains, stresses,
                                           dstress_dmat_list, *aim_args)

    # Sensitivity analysis
    deriv = sensitivity_analysis_per_grid_pts(cell, krylov_solver, strains,
                                 deriv_strains, deriv_phase, dstress_dmat_list, tol)
    deriv = deriv.reshape(nb_grid_pts, order='F')

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        # Iterate over pixels
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                # Disturb material
                phase[i, j] += delta
                density = node_to_quad_pt_2_quad_pts_sequential(phase)
                density = density.reshape([nb_quad_pts, nb_pixels], order='F')
                lame1 = delta_lame1 * density ** order
                lame2 = delta_lame2 * density ** order
                for pixel_id, pixels in cell.pixels.enumerate():
                    for quad_id in range(cell.nb_quad_pts):
                        quad = nb_quad_pts * pixel_id + quad_id
                        mat.set_lame_constants(quad, lame1[quad_id, pixel_id],
                                               lame2[quad_id, pixel_id])

                # Calculate new strains + stresses
                stresses = []
                strains = []
                for DelF in DelFs:
                    r = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                            tol, tol, verbose)
                    stress = r.stress.copy()
                    stresses.append(stress)
                    strain = r.grad.copy()
                    strains.append(strain)

                # New aim function
                aim_plus = aim_function(cell, phase, strains, stresses, *aim_args)
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
        fig.suptitle('2 quad pts and 2 strains')
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
    test_sensitivity_only_phase_field(plot=True)
    test_sensitivity_analysis_per_quad_pts(plot=True)
    test_sensitivity_analysis_per_grid_pts(plot=True)
    test_sensitivity_analysis_per_grid_pts_two(plot=True)
