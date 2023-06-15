"""
Functions and code snippets controlling the overall calculation, including
calling muSpectre.
"""
import os
import numpy as np
import muSpectre as µ
from muSpectre import sensitivity_analysis as sa
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muTopOpt.Filter import map_to_unit_range
from muTopOpt.Filter import map_to_unit_range_derivative
from muTopOpt.PhaseField import phase_field_rectangular_grid_deriv_phase
from muTopOpt.PhaseField import phase_field_rectangular_grid
from muTopOpt.PhaseField import phase_field_parallelogram_grid_deriv_phase
from muTopOpt.PhaseField import phase_field_parallelogram_grid
from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.StressTarget import square_error_target_stresses_deriv_strains
from muTopOpt.StressTarget import square_error_target_stresses_deriv_phase
#import muTopOpt.sensitivity_analysis as sa


### ----- For rectangle grid ----- ###
def aim_function(phase, strains, stresses, cell, args):
    """ Calculate the aim function for the optimization.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    aim: float
         Value of aim function.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    aim = square_error_target_stresses(cell, strains, stresses, target_stresses)
    aim += weight * phase_field_rectangular_grid(phase, eta, cell)
    return aim

def aim_function_deriv_strains(phase, strains, stresses, cell, args):
    """ Calculate the partial derivative of the aim function with respect to the strains.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: list of np.ndarrays(dim**2 * nb_quad_pts * nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the strains.
    """
    target_stresses = args[0]
    derivatives = square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses)
    return derivatives

def aim_function_deriv_phase(phase, strains, stresses, cell, Young, delta_Young, Poisson,
                             delta_Poisson, dstress_dphase_list, args):
    """ Calculate the partial derivative of the aim function with respect to the design parameters.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    Young: np.ndarray(nb_pixels) of floats
           Youngs modulus for each pixel
    delta_Young: float
                 Max difference of Youngs modulus.
    Poisson: np.ndarray(nb_pixels) of floats
             Poissons ratio for each pixel
    delta_Poisson: float
                   Max difference of Poissons ratio.
    dstress_dphase: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the strains.
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: np.ndarray(nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the design parameters.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    derivatives = square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                                           dstress_dphase_list)
    derivatives *= map_to_unit_range_derivative(phase)
    derivatives += weight * phase_field_rectangular_grid_deriv_phase(phase, eta, cell)
    return derivatives

def call_function(phase, cell, mat, Young1, Poisson1, Young2, Poisson2, DelFs,
                  nb_strain_steps, krylov_solver_args, solver_args, args, gradient=None, weights=None, calc_sens=True,
                  file_tmp=None, file_last=None, file_evo=None, file_evo_details=None, verbose=False):
    """ Calculate the aim function and the sensitivity.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    cell: object
          muSpectre cell object
    mat: object
         muSpectre cell object belonging to cell
    Young1: float
            Youngs modulus for phase=0
    Poisson1:float
             Poissons ratio for phase=0
    Young2: float
            Youngs modulus for phase=1
    Poisson2: float
             Poissons ratio for phase=1
    DelFs: list of np.ndarray(dim, dim) of floats
        List of prescribed macroscopic strain
    nb_strain_steps: int
        The prescribed macroscopic strains are applied in nb_strain_steps
        uniform intervalls.
    krylov_solver_args: list
        List of additional arguments passed to the krylov_solver
    solver_args: list
        List of additional arguments passed to the solver
    args: list
        list with additional arguments passed to the aim function
    gradient: list
              Contains the stencils for the discrete derivation operator. Default is None.
    weights: list of floats
             Weights for the quadrature points. Default is None.
    calc_sens: boolean
               If False, the sensitivity is not calculated. Default is True.
    file_tmp: string
              .npy-file for saving the new phase before the muSpectre calculation.
              After the muSpectre calculation, the file is deleted.
    file_last: string
               .npy-file for saving the phase of the last successful optimization step
    file_evo: string
             .txt file in which the last aim function is added
    file_evo_details: string
                      .txt file in which the last aim function and some details are added
    verbose: boolean
             If True, the last aim function is printed. Default is false.
    Returns
    -------
    aim: float
         Aim function
    S: np.ndarray(nb_pixels) of floats
       Sensitivity
    """
    if file_tmp is not None:
        phase = phase.reshape(*cell.nb_subdomain_grid_pts, order='F')
        save_npy(file_tmp, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    phase = phase.flatten(order='F')

    # Change material of cell
    Young = (Young2 - Young1) * map_to_unit_range(phase) + Young1
    Poisson = (Poisson2 - Poisson1) * map_to_unit_range(phase) + Poisson1
    for pixel_id, pixel in cell.pixels.enumerate():
        quad_id = cell.nb_quad_pts * pixel_id
        for i in range(cell.nb_quad_pts):
            mat.set_youngs_modulus_and_poisson_ratio(quad_id + i, Young[pixel_id], Poisson[pixel_id])

    # Solve the equilibrium equations
    dim = cell.dim
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    stresses = []
    for DelF in DelFs:
        applied_strain = []
        for s in range(1, nb_strain_steps+1):
            applied_strain.append(s / nb_strain_steps * DelF)
        result = µ.solvers.newton_cg(cell, applied_strain, krylov_solver,
                                     *solver_args)
        strain = result[nb_strain_steps-1].grad.reshape(shape, order='F').copy()
        strains.append(strain)
        stresses.append(cell.evaluate_stress(strain).copy())

    # Calculate the aim function
    aim = aim_function(phase, strains, stresses, cell, args)

    # Calculate the sensitivity
    if calc_sens:
        S = sa.sensitivity_analysis(aim_function_deriv_strains, aim_function_deriv_phase,
                                    phase, Young1, Poisson1, Young2, Poisson2, cell, krylov_solver,
                                    strains, stresses, gradient=gradient, weights=weights,
                                    args=args, filter_func=map_to_unit_range,
                                    dfilter_dphase=map_to_unit_range_derivative)

    # Remove file with temporary phase if muSpectre calculations worked
    if MPI.COMM_WORLD.rank == 0:
        if file_tmp is not None:
            os.remove(file_tmp)

    # Save the last step
    if file_last is not None:
        phase = phase.reshape(*cell.nb_subdomain_grid_pts, order='F')
        save_npy(file_last, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    if (MPI.COMM_WORLD.rank == 0) and (file_evo is not None):
        with open(file_evo, 'a') as f:
            print(aim, file=f)
    if file_evo_details is not None:
        sq_err = square_error_target_stresses(cell, strains, stresses, args[0])
        ph_field = args[1] * phase_field_rectangular_grid(phase, args[2], cell)
        norm_S = np.linalg.norm(S)**2
        norm_S = np.sqrt(Reduction(MPI.COMM_WORLD).sum(norm_S))
        if (MPI.COMM_WORLD.rank == 0):
            with open(file_evo_details, 'a') as f:
                print(aim, sq_err, ph_field, norm_S, file=f)

    if calc_sens:
        return aim, S.flatten(order='F')
    else:
        return aim

### ----- For parallelogram grid ----- ###
def aim_function_parall(phase, strains, stresses, cell, args):
    """ Calculate the aim function for the optimization.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Phase field function.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    aim: float
         Value of aim function.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    aim = square_error_target_stresses(cell, strains, stresses, target_stresses)
    aim += weight * phase_field_parallelogram_grid(phase, eta, cell)
    return aim

def aim_function_parall_deriv_strains(phase, strains, stresses, cell, args):
    """ Calculate the partial derivative of the aim function with respect to the strains.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: list of np.ndarrays(dim**2 * nb_quad_pts * nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the strains.
    """
    target_stresses = args[0]
    derivatives = square_error_target_stresses_deriv_strains(cell, strains, stresses, target_stresses)
    return derivatives

def aim_function_parall_deriv_phase(phase, strains, stresses, cell, Young, delta_Young, Poisson,
                             delta_Poisson, dstress_dphase_list, args):
    """ Calculate the partial derivative of the aim function with respect to the design parameters.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
             List of microscopic strains. Must have the same length as target_stresses.
    stresses: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
              List of microscopic stresses. Must have the same length as target_stresses.
    cell: object
          muSpectre cell object
    Young: np.ndarray(nb_pixels) of floats
           Youngs modulus for each pixel
    delta_Young: float
                 Max difference of Youngs modulus.
    Poisson: np.ndarray(nb_pixels) of floats
             Poissons ratio for each pixel
    delta_Poisson: float
                   Max difference of Poissons ratio.
    dstress_dphase: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the strains.
    args: list with additional arguments
        Contains:
        target_stresses: list of np.ndarray(dim, dim) of floats
                         List of target stresses. Must have the same length as stresses.
        weight: float
                Weighting parameter between the phase field and the target stresses.
        eta: float
             Weighting parameter of the phase field energy. A larger eta means a broader interface.
    Returns
    -------
    derivatives: np.ndarray(nb_pixels) of floats
                 Partial derivatives of the aim function with respect to the design parameters.
    """
    target_stresses = args[0]
    weight = args[1]
    eta = args[2]
    derivatives = square_error_target_stresses_deriv_phase(cell, stresses, target_stresses,
                                                           dstress_dphase_list)
    derivatives *= map_to_unit_range_derivative(phase)
    derivatives += weight * phase_field_parallelogram_grid_deriv_phase(phase, eta, cell)
    return derivatives

def call_function_parall(phase, cell, mat, Young1, Poisson1, Young2, Poisson2, DelFs,
                  nb_strain_steps, krylov_solver_args, solver_args, args, gradient=None, weights=None, calc_sens=True,
                  file_tmp=None, file_last=None, file_evo=None, file_evo_details=None, verbose=False):
    """ Calculate the aim function and the sensitivity.

    Parameters
    ----------
    phase: np.ndarray(nb_pixels) of floats
           Design parameters.
    cell: object
          muSpectre cell object
    mat: object
         muSpectre cell object belonging to cell
    Young1: float
            Youngs modulus for phase=0
    Poisson1:float
             Poissons ratio for phase=0
    Young2: float
            Youngs modulus for phase=1
    Poisson2: float
             Poissons ratio for phase=1
    DelFs: list of np.ndarray(dim, dim) of floats
        List of prescribed macroscopic strain
    nb_strain_steps: int
        The prescribed macroscopic strains are applied in nb_strain_steps
        uniform intervalls.
    krylov_solver_args: list
        List of additional arguments passed to the krylov_solver
    solver_args: list
        List of additional arguments passed to the solver
    args: list
        list with additional arguments passed to the aim function
    gradient: list
              Contains the stencils for the discrete derivation operator. Default is None.
    weights: list of floats
             Weights for the quadrature points. Default is None.
    calc_sens: boolean
               If False, the sensitivity is not calculated. Default is True.
    file_tmp: string
              .npy-file for saving the new phase before the muSpectre calculation.
              After the muSpectre calculation, the file is deleted.
    file_last: string
               .npy-file for saving the phase of the last successful optimization step
    file_evo: string
             .txt file in which the last aim function is added
    file_evo_details: string
                      .txt file in which the last aim function and some details are added
    verbose: boolean
             If True, the last aim function is printed. Default is false.
    Returns
    -------
    aim: float
         Aim function
    S: np.ndarray(nb_pixels) of floats
       Sensitivity
    """
    if file_tmp is not None:
        phase = phase.reshape(*cell.nb_subdomain_grid_pts, order='F')
        save_npy(file_tmp, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    phase = phase.flatten(order='F')

    # Change material of cell
    Young = (Young2 - Young1) * map_to_unit_range(phase) + Young1
    Poisson = (Poisson2 - Poisson1) * map_to_unit_range(phase) + Poisson1
    for pixel_id, pixel in cell.pixels.enumerate():
        quad_id = cell.nb_quad_pts * pixel_id
        for i in range(cell.nb_quad_pts):
            mat.set_youngs_modulus_and_poisson_ratio(quad_id + i, Young[pixel_id], Poisson[pixel_id])

    # Solve the equilibrium equations
    dim = cell.dim
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    stresses = []
    for DelF in DelFs:
        applied_strain = []
        for s in range(1, nb_strain_steps+1):
            applied_strain.append(s / nb_strain_steps * DelF)
        result = µ.solvers.newton_cg(cell, applied_strain, krylov_solver,
                                     *solver_args)
        strain = result[nb_strain_steps-1].grad.reshape(shape, order='F').copy()
        strains.append(strain)
        stresses.append(cell.evaluate_stress(strain).copy())

    # Calculate the aim function
    aim = aim_function_parall(phase, strains, stresses, cell, args)

    # Calculate the sensitivity
    if calc_sens:
        S = sa.sensitivity_analysis(aim_function_parall_deriv_strains, aim_function_parall_deriv_phase,
                                    phase, Young1, Poisson1, Young2, Poisson2, cell, krylov_solver,
                                    strains, stresses, gradient=gradient, weights=weights,
                                    args=args, filter_func=map_to_unit_range,
                                    dfilter_dphase=map_to_unit_range_derivative)

    # Remove file with temporary phase if muSpectre calculations worked
    if MPI.COMM_WORLD.rank == 0:
        if file_tmp is not None:
            os.remove(file_tmp)

    # Save the last step
    if file_last is not None:
        phase = phase.reshape(*cell.nb_subdomain_grid_pts, order='F')
        save_npy(file_last, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
    if (MPI.COMM_WORLD.rank == 0) and (file_evo is not None):
        with open(file_evo, 'a') as f:
            print(aim, file=f)
    if file_evo_details is not None:
        sq_err = square_error_target_stresses(cell, strains, stresses, args[0])
        ph_field = args[1] * phase_field_parallelogram_grid(phase, args[2], cell)
        norm_S = np.linalg.norm(S)**2
        norm_S = np.sqrt(Reduction(MPI.COMM_WORLD).sum(norm_S))
        if (MPI.COMM_WORLD.rank == 0):
            with open(file_evo_details, 'a') as f:
                print(aim, sq_err, ph_field, norm_S, file=f)

    if calc_sens:
        return aim, S.flatten(order='F')
    else:
        return aim
