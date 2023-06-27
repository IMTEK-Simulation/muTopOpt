"""
Tests the complete sensitivity calculation
"""
import numpy as np
import muSpectre as µ
from muSpectre import sensitivity_analysis as sa
import muFFT

from muTopOpt.Controller import call_function
from muTopOpt.Controller import call_function_parall
from muTopOpt.Controller import aim_function
from muTopOpt.Controller import aim_function_deriv_strains
from muTopOpt.Controller import aim_function_deriv_phase
from muTopOpt.Filter import map_to_unit_range
from muTopOpt.PhaseField import phase_field_rectangular_grid
from muTopOpt.PhaseField import phase_field_rectangular_grid_deriv_phase
from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.StressTarget import square_error_target_stresses_deriv_strains
from muTopOpt.StressTarget import square_error_target_stresses_deriv_phase


###########################################################################
### -------------------- Test only phase-field term ------------------- ###
###########################################################################
def test_sensitivity_only_phase_field(plot=False):
    ### ----- Aim function + partial derivatives ----- ###
    def aim_phase(phase, strains, stresses, cell, args):
        return phase_field_rectangular_grid(phase, args[2], cell)

    def aim_phase_deriv_strains(phase, strains, stresses, cell, args):
        return len(strains) * [np.zeros(strains[0].shape)]

    def aim_phase_deriv_phase(phase, strains, stresses, cell, Young,
                              delta_Young, Poisson, delta_Poisson,
                              dstress_dphase_list, args):
        return phase_field_rectangular_grid_deriv_phase(phase, args[2], cell)

    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

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

    # Weighting parameters
    weight = 0
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalize
        stress = stress / loading / Young2
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material initialization
    Young = (Young2 - Young1) * map_to_unit_range(phase) + Young1
    Poisson = (Poisson2 - Poisson1) * map_to_unit_range(phase) + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    stresses = []
    for DelF in DelFs:
        result = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                     *solver_args)
        strain = result.grad.reshape(shape, order='F').copy()
        strains.append(strain)
        stresses.append(cell.evaluate_stress(strain).copy())

    # Calculate aim function
    aim = aim_phase(phase, strains, stresses, cell, args)

    # Calculate derivative
    strains = []
    stresses = []
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    for DelF in DelFs:
        res = µ.solvers.newton_cg(cell, DelF, krylov_solver, *solver_args)
        strains.append(res.grad.copy())
        stresses.append(res.stress.copy())
    dstress_dphase_list = sa.calculate_dstress_dphase(cell, strains, Young, Young2 - Young1, Poisson,
                              Poisson2 - Poisson1, gradient=None, weights=None)
    deriv = sa.sensitivity_analysis(aim_phase_deriv_strains, aim_phase_deriv_phase, phase,
                                    Young1, Poisson1, Young2, Poisson2, cell,
                                    krylov_solver, strains, stresses, args=args)
    deriv = deriv.flatten(order='F')

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(phase)):
            phase[i] += delta
            strains = []
            stresses = []
            for DelF in DelFs:
                result = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                             *solver_args)
                strain = result.grad.reshape(shape, order='F').copy()
                strains.append(strain)
                stresses.append(cell.evaluate_stress(strain).copy())
            aim_plus = aim_phase(phase, strains, stresses, cell, args)
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

###########################################################################
### ----------------- Test only target stresses term ------------------ ###
###########################################################################
def test_sensitivity_target_stresses_no_filter(plot=False):
    """ Test the sensitivity of the target stresses without
        a filter function.
    """
    ### ----- Aim function + partial derivatives ----- ###
    def aim_no_filter(phase, strains, stresses, cell, args):
        return square_error_target_stresses(cell, strains, stresses, args[0],
                                            args[3], args[4])

    def aim_no_filter_deriv_strains(phase, strains, stresses, cell, args):
        return square_error_target_stresses_deriv_strains(cell, strains,
                                                          stresses, args[0],
                                                          args[3], args[4])

    def aim_no_filter_deriv_phase(phase, strains, stresses, cell, Young,
                              delta_Young, Poisson, delta_Poisson,
                              dstress_dphase_list, args):
        return square_error_target_stresses_deriv_phase(cell, stresses, args[0],
                                                        dstress_dphase_list,
                                                        args[3], args[4])

    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.7]
    formulation = µ.Formulation.small_strain

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.35

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.011
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

    # Weighting parameters
    weight = 0
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material initialization
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    stresses = []
    for DelF in DelFs:
        result = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                     *solver_args)
        strain = result.grad.reshape(shape, order='F').copy()
        strains.append(strain)
        stresses.append(cell.evaluate_stress(strain).copy())

    # Calculate aim function
    aim = aim_no_filter(phase, strains, stresses, cell, args)

    # Calculate derivative
    strains = []
    stresses = []
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    for DelF in DelFs:
        res = µ.solvers.newton_cg(cell, DelF, krylov_solver, *solver_args)
        strains.append(res.grad.copy())
        stresses.append(res.stress.copy())
    dstress_dphase_list = sa.calculate_dstress_dphase(cell, strains, Young, Young2 - Young1, Poisson,
                              Poisson2 - Poisson1, gradient=None, weights=None)
    deriv = sa.sensitivity_analysis(aim_no_filter_deriv_strains, aim_no_filter_deriv_phase, phase,
                                    Young1, Poisson1, Young2, Poisson2, cell,
                                    krylov_solver, strains, stresses, args=args)
    deriv = deriv.flatten(order='F')

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(len(phase)):
            phase[i] += delta
            Young = (Young2 - Young1) * phase + Young1
            Poisson = (Poisson2 - Poisson1) * phase + Poisson1
            for pixel_id, pixel in cell.pixels.enumerate():
                quad_id = cell.nb_quad_pts * pixel_id
                for i_quad in range(cell.nb_quad_pts):
                    mat.set_youngs_modulus_and_poisson_ratio(quad_id + i_quad,
                                                             Young[pixel_id],
                                                             Poisson[pixel_id])
            strains = []
            stresses = []
            for DelF in DelFs:
                result = µ.solvers.newton_cg(cell, DelF, krylov_solver,
                                             *solver_args)
                strain = result.grad.reshape(shape, order='F').copy()
                strains.append(strain)
                stresses.append(cell.evaluate_stress(strain).copy())
            aim_plus = aim_no_filter(phase, strains, stresses, cell, args)
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
        ax.set_ylabel('Abs error of sensitivity (only target stresses)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-4

def test_sensitivity_target_stresses_filter(plot=False):
    """ Test the sensitivity of the target stresses with
        a filter function.
    """
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = None
    weights = None

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.009
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

    # Weighting parameters
    weight = 0
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # Calculate aim function + sensitivity
    aim, S = call_function(phase, cell, mat, Young1, Poisson1,
                           Young2, Poisson2, DelFs, nb_strain_steps,
                           krylov_solver_args, solver_args, args, calc_sens=True,
                           gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        S_fin_diff = np.empty(S.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = call_function(phase, cell, mat, Young1, Poisson1,
                                     Young2, Poisson2, DelFs, nb_strain_steps,
                                     krylov_solver_args, solver_args, args,
                                     calc_sens=False, gradient=gradient,
                                     weights=weights)
            S_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

        diff = np.linalg.norm(S_fin_diff - S)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity (target stresses + filter)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-5

###########################################################################
### --------------- Test complete sensitivity analysis ---------------- ###
###########################################################################
def test_sensitivity_analysis_complete(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = None
    weights = None

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.009
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

    # Weighting parameters
    weight = 0.3
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # Calculate aim function + sensitivity
    aim, S = call_function(phase, cell, mat, Young1, Poisson1,
                           Young2, Poisson2, DelFs, nb_strain_steps,
                           krylov_solver_args, solver_args, args, calc_sens=True, gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        S_fin_diff = np.empty(S.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = call_function(phase, cell, mat, Young1, Poisson1,
                                     Young2, Poisson2, DelFs, nb_strain_steps,
                                     krylov_solver_args, solver_args, args, calc_sens=False,
                                     gradient=gradient, weights=weights)
            S_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

        diff = np.linalg.norm(S_fin_diff - S)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity (complete)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-4


def test_sensitivity_analysis_complete_two_quad(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    weights=[1, 1]

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.009
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

    # Weighting parameters
    weight = 0.3
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # Calculate aim function + sensitivity
    aim, S = call_function(phase, cell, mat, Young1, Poisson1,
                           Young2, Poisson2, DelFs, nb_strain_steps,
                           krylov_solver_args, solver_args, args, calc_sens=True, gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        S_fin_diff = np.empty(S.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = call_function(phase, cell, mat, Young1, Poisson1,
                                Young2, Poisson2, DelFs, nb_strain_steps,
                                krylov_solver_args, solver_args, args, calc_sens=False, gradient=gradient, weights=weights)
            S_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

        diff = np.linalg.norm(S_fin_diff - S)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity (2 quad pts)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-4

def test_sensitivity_analysis_complete_one_stress_entry(plot=False):
    ### ----- Set up ----- ###
    # Consider only the xy-direction of the stress tensor
    case_stress_entry = 3

    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    weights=[1, 1]

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.009
    DelFs[0][0, 1] = loading / 2
    DelFs[0][1, 0] = loading / 2
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2
    nb_strain_steps = 1

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (tol, maxiter, verbose)
    solver_args = (tol, tol, verbose)

    # Weighting parameters
    weight = 0.3
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2, case_stress_entry)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # Calculate aim function + sensitivity
    aim, S = call_function(phase, cell, mat, Young1, Poisson1,
                           Young2, Poisson2, DelFs, nb_strain_steps,
                           krylov_solver_args, solver_args, args, calc_sens=True,
                           gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        S_fin_diff = np.empty(S.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = call_function(phase, cell, mat, Young1, Poisson1,
                                     Young2, Poisson2, DelFs, nb_strain_steps,
                                     krylov_solver_args, solver_args, args, calc_sens=False,
                                     gradient=gradient, weights=weights)
            S_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

        diff = np.linalg.norm(S_fin_diff - S)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity (2 quad pts)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-5


###########################################################################
### ------------------- Test other discretization --------------------- ###
###########################################################################
def test_sensitivity_analysis_parallelograms(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    weights=[1, 1]

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    Young1 = 0
    Young2 = 12
    Poisson1 = 0
    Poisson2= 0.3

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    loading = 0.009
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

    # Weighting parameters
    weight = 0.3
    eta = 0.025

    # List of finite differences
    if plot:
        delta_list = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Calculate target stresses
    Young_av = (Young2 - Young1) / 2
    Poisson_av = (Poisson2 - Poisson1) / 2
    mu = 0.5 * Young_av / (1 + Poisson_av)
    lam = Poisson_av / (1 - 2 * Poisson_av) * 0.5 * Young_av / (1 + Poisson_av)
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * mu * DelF + lam * np.trace(DelF) * np.eye(dim)
        # Nondimensionalization
        stress = stress / Young2 / loading
        target_stresses.append(stress)
    args = (target_stresses, weight, eta, loading, Young2)

    # Initialize cell
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights)
    Young = (Young2 - Young1) * phase + Young1
    Poisson = (Poisson2 - Poisson1) * phase + Poisson1
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
    cell.initialise()

    # Calculate aim function + sensitivity
    aim, S = call_function_parall(phase, cell, mat, Young1, Poisson1,
                           Young2, Poisson2, DelFs, nb_strain_steps,
                           krylov_solver_args, solver_args, args, calc_sens=True, gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        S_fin_diff = np.empty(S.shape)
        for i in range(len(phase)):
            phase[i] += delta
            aim_plus = call_function_parall(phase, cell, mat, Young1, Poisson1,
                                Young2, Poisson2, DelFs, nb_strain_steps,
                                krylov_solver_args, solver_args, args, calc_sens=False, gradient=gradient, weights=weights)
            S_fin_diff[i] = (aim_plus - aim) / delta
            phase[i] -= delta

        diff = np.linalg.norm(S_fin_diff - S)
        diff_list.append(diff)

    ### ----- Fit ----- ###
    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of sensitivity (2 quad pts)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 5e-5
