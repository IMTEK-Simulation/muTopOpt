"""
Tests the StressTarget functions
"""
import numpy as np
import muSpectre as µ

from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.StressTarget import square_error_target_stresses_deriv_strains
from muTopOpt.StressTarget import square_error_target_stresses_deriv_phase

def test_square_error_target_stresses():
    """ Check the implementation of the target stresses term.
    """
    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    Young = 10.
    Poisson = 0.3
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id, Young, Poisson)

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    ### ----- Calculate stresses ----- ###
    # Lame constants
    mu = 0.5 * Young / (1 + Poisson)
    lam = Poisson / (1 - 2 * Poisson) * Young / (1 + Poisson)

    # Analytical stresses for homogenous material
    ana_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        ana_stresses.append(stress)

    # muSpectre calculations
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    ### ----- Test square_error_target_stresses() ----- ###
    sq_error = square_error_target_stresses(cell, strains, stresses, ana_stresses)

    assert True

def test_square_error_target_stresses_deriv_strain(plot=False):
    """ Check the implementation of the derivative of the target stresses
        term with respect to the strains on a rectangular grid for one load case.
    """

    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young = 10.
    Poisson = 0.3

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

    ### ----- Target stresses ----- ###
    # Stresses for homogenous material with half the Youngs modulus
    mu = 0.5 * 0.5 * Young / (1 + Poisson)
    lam = Poisson / (1 - 2 * Poisson) * 0.5 * Young / (1 + Poisson)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    Young = Young * phase
    Poisson = Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

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
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses)
    derivs = square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses)

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
                helper_stresses[i_case] = cell.evaluate_stress(strain.reshape(shape, order='F'))
                sq_err_plus = square_error_target_stresses(cell, strains, helper_stresses, target_stresses)
                deriv_fin_diff[i] = (sq_err_plus - sq_err) / delta
                strain[i] -= delta
            helper_stresses[i_case]
            diff += (deriv_fin_diff - deriv)
        diff = np.linalg.norm(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6

def test_square_error_target_stresses_deriv_strain_two(plot=False):
    """ Check the implementation of the derivative of the target stresses
        term with respect to the strains on a rectangular grid for two load cases.
    """

    ### ----- Set-up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young = 10.
    Poisson = 0.3

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
        # delta_list = [1e-5]
    else:
        delta_list = [1e-4, 5e-5]

    ### ----- Target stresses ----- ###
    # Stresses for homogenous material with half the Youngs modulus
    mu = 0.5 * 0.5 * Young / (1 + Poisson)
    lam = Poisson / (1 - 2 * Poisson) * 0.5 * Young / (1 + Poisson)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    Young = Young * phase
    Poisson = Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

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
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses)
    derivs = square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses)

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
                helper_stresses[i_case] = cell.evaluate_stress(strain.reshape(shape, order='F'))
                sq_err_plus = square_error_target_stresses(cell, strains, helper_stresses, target_stresses)
                deriv_fin_diff[i] = (sq_err_plus - sq_err) / delta
                strain[i] -= delta
            helper_stresses[i_case] = stresses[i_case].copy()
            diff += (deriv_fin_diff - deriv)
        diff = np.linalg.norm(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6
