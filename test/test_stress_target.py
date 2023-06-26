"""
Tests the StressTarget functions
"""
import numpy as np
import muSpectre as µ
from muSpectre import sensitivity_analysis as sa

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
    Young = 11.
    Poisson = 0.3
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
            mat.add_pixel(pixel_id, Young, Poisson)

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.02
    DelFs[0][0, 0] = loading
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
        # Nondimensionalize
        stress = stress / loading / Young
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
    sq_error = square_error_target_stresses(cell, strains, stresses,
                                            ana_stresses, loading, Young)

    assert abs(sq_error) < 1e-7

def test_square_error_target_stresses_deriv_strain(plot=False):
    """ Check the implementation of the derivative of the target stresses
        term with respect to the strains for one load case.
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
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young0 = 12.
    Poisson0 = 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim])]
    loading = 0.02
    DelFs[0][0, 0] = loading

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
    Young_av = Young0 / 2
    Poisson_av = Poisson0 / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalize
        stress = stress / loading / Young0
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    Young = Young0 * phase
    Poisson = Poisson0 * phase
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
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses,
                                          loading, Young0)
    derivs = square_error_target_stresses_deriv_strains(cell, strains, stresses,
                                                        target_stresses, loading, Young0)

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
                sq_err_plus = square_error_target_stresses(cell, strains, helper_stresses,
                                                           target_stresses, loading, Young0)
                deriv_fin_diff[i] = (sq_err_plus - sq_err) / delta
                strain[i] -= delta
            helper_stresses[i_case]
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

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6

def test_square_error_target_stresses_deriv_strain_two(plot=False):
    """ Check the implementation of the derivative of the target stresses
        term with respect to the strains for two load cases.
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
    Young0 = 12.
    Poisson0 = 0.3

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
    # Stresses for homogenous material with half the Youngs modulus
    mu = 0.5 * 0.5 * Young0 / (1 + Poisson0)
    lam = Poisson0 / (1 - 2 * Poisson0) * 0.5 * Young0 / (1 + Poisson0)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalize
        stress = stress / loading / Young0
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    Young = Young0 * phase
    Poisson = Poisson0 * phase
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
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses,
                                          loading, Young0)
    derivs = square_error_target_stresses_deriv_strains(cell, strains, stresses,
                                                        target_stresses, loading, Young0)

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
                sq_err_plus = square_error_target_stresses(cell, strains, helper_stresses,
                                                           target_stresses, loading, Young0)
                deriv_fin_diff[i] = (sq_err_plus - sq_err) / delta
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
        ax.set_ylabel('Abs error of partial deriv StressTarget w.r.t. strains (two strains)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6


def test_square_error_target_stresses_deriv_phase(plot=False):
    """ Check the implementation of the derivative of the target stresses
        term with respect to the phase for two load cases.
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
    delta_Young = 12.
    delta_Poisson = 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.011
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
    Young_av = delta_Young / 2
    Poisson_av = delta_Poisson / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalize
        stress = stress / loading / delta_Young
        target_stresses.append(stress)

    ### ----- Analytical derivative ----- ###
    # Material initialization
    Young = delta_Young * phase
    Poisson = delta_Poisson * phase
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
        strain = r.grad.copy()
        strains.append(strain)

    dstress_dphase_list = sa.calculate_dstress_dphase(cell, strains, Young,
                                                      delta_Young, Poisson,
                                                      delta_Poisson)

    # Function and derivative
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses,
                                          loading, delta_Young)
    deriv = square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                                     dstress_dphase_list, loading,
                                                     delta_Young)

    ### ----- Finite difference derivatives ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(deriv)):
            # Cell with disturbed material
            phase[i] += delta
            Young = delta_Young * phase
            Poisson = delta_Poisson * phase
            for pixel_id, pixel in cell.pixels.enumerate():
                quad_id = cell.nb_quad_pts * pixel_id
                for i_quad in range(cell.nb_quad_pts):
                    mat.set_youngs_modulus_and_poisson_ratio(quad_id + i_quad,
                                                             Young[pixel_id],
                                                             Poisson[pixel_id])
            # New stresses
            stresses = []
            for strain in strains:
                stress = cell.evaluate_stress(strain.reshape(shape, order='F'))
                stresses.append(stress.copy())

            # Derivative of square_error_target_stresses()
            sq_err_plus = square_error_target_stresses(cell, strains, stresses,
                                                       target_stresses, loading, delta_Young)
            deriv_fin_diff[i] = (sq_err_plus - sq_err) / delta
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
        ax.set_ylabel('Abs error of partial deriv StressTarget w.r.t. phase')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6
