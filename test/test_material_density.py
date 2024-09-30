"""
Tests the filter functions
"""
import sys
import os
import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ

from muTopOpt.MaterialDensity import map_to_unit_range
from muTopOpt.MaterialDensity import map_to_unit_range_derivative
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative_sequential
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts
# from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative

from muTopOpt.Controller import calculate_dstress_dmat
from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.StressTarget import square_error_target_stresses_deriv_phase

def test_map_to_unit_range_derivative(plot=False):
    """ Check the implementation of the derivative of the filter function.
    """
    # Set up
    nb_grid_pts = [5, 4]
    arr = 2 * np.random.random(nb_grid_pts) - 0.5
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    # Analytical derivative
    func = map_to_unit_range(arr)
    deriv = map_to_unit_range_derivative(arr)

    # Finite difference derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                arr[i, j] += delta
                func_plus = map_to_unit_range(arr)
                deriv_fin_diff[i, j] = (func_plus[i, j] - func[i, j]) / delta
                arr[i, j] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of derivative of filter-function')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_node_to_quad_pt_2_quad_pts():
    """ Test the interpolation of a field from
        the nodal values to the quadrature points.
    """
    # Setup
    nb_grid_pts = [3, 5]
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    phase = np.random.random(nb_grid_pts)
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    material = node_to_quad_pt_2_quad_pts_sequential(phase)

    # Test for first pixel - first quadrature point
    a = material[0, 1, 1]
    b = (phase[1, 1] + phase[1, 2] + phase[2, 1]) / 3
    assert abs(a - b) < 1e-7

    # Test for first pixel - second quadrature point
    a = material[1, 1, 1]
    b = (phase[1, 2] + phase[2, 2] + phase[2, 1]) / 3
    assert abs(a - b) < 1e-7

    # Test for second pixel - first quadrature point
    a = material[0, 2, 0]
    b = (phase[2, 0] + phase[2, 1] + phase[0, 0]) / 3
    assert abs(a - b) < 1e-7

    # Test for second pixel - second quadrature point
    a = material[1, 2, 0]
    b = (phase[0, 0] + phase[0, 1] + phase[2, 1]) / 3
    assert abs(a - b) < 1e-7

    # Test parallel implementation
    material2 = node_to_quad_pt_2_quad_pts(phase, cell)

    assert np.linalg.norm(material - material2) < 1e-7

def test_df_dphase_2_quad_pts_derivative(plot=False):
    """ Test the derivative of df_dphase_2_quad_pts_derivative.
    """
    def func(material):
        return np.sum(material**2)
    def func_deriv_mat(material):
        return 2 * material.copy()

    nb_grid_pts = [3, 3]
    phase = np.random.random(nb_grid_pts)
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    else:
        delta_list = [1e-4, 5e-5]

    # Analytical calculation of derivative
    material = node_to_quad_pt_2_quad_pts_sequential(phase)
    aim = func(material)
    deriv_mat = func_deriv_mat(material)
    deriv_mat = deriv_mat.reshape([-1, 2, *nb_grid_pts], order='F')
    deriv_ana =\
        df_dphase_2_quad_pts_derivative_sequential(deriv_mat)
    deriv_ana = deriv_ana.reshape(nb_grid_pts)

    # Finite difference calculation of derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(nb_grid_pts)
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                phase[i, j] += delta
                material = node_to_quad_pt_2_quad_pts_sequential(phase)
                aim_plus = func(material)
                deriv_fin_diff[i, j] = (aim_plus - aim) / delta
                phase[i, j] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv_ana)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of derivative of filter-function')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_df_dphase_2_quad_pts_derivative_2(plot=False):
    """ Test the derivative of df_dphase_2_quad_pts_derivative
        for f = StressTarget.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [2, 5]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)

    # Material
    np.random.seed(1)
    phase = np.random.random(nb_grid_pts)
    delta_lame1 = 0.3
    delta_lame2 = 12
    order = 2

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

    density = density.reshape([cell.nb_quad_pts, *nb_grid_pts], order='F')
    dstress_dmat_list = calculate_dstress_dmat(cell, mat, strains, density,
                                                 delta_lame1, delta_lame2,
                                                 0, 0, order=order)
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses)
    deriv_mat = square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                                         dstress_dmat_list)
    deriv_mat = deriv_mat.reshape([-1, 2, *nb_grid_pts], order='F')
    deriv_phase =\
        df_dphase_2_quad_pts_derivative_sequential(deriv_mat)
    deriv_phase = deriv_phase.reshape(nb_grid_pts)

    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv_phase.shape)
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
                        mat.set_lame_constants(quad_id, lame1[q, pixel_id], lame2[q, pixel_id])
                # New stresses
                stresses = []
                for strain in strains:
                    stress = cell.evaluate_stress(strain.reshape(shape, order='F'))
                    stresses.append(stress.copy())
                # Derivative of aim function
                sq_err_plus = square_error_target_stresses(cell, strains,
                                                           stresses, target_stresses)
                deriv_fin_diff[i, j] = (sq_err_plus - sq_err) / delta
                phase[i, j] -= delta
        diff = np.linalg.norm(deriv_fin_diff - deriv_phase)
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
    test_map_to_unit_range_derivative(plot=False)
    test_node_to_quad_pt_2_quad_pts()
    test_df_dphase_2_quad_pts_derivative(plot=True)
    test_df_dphase_2_quad_pts_derivative_2(plot=True)
