"""
Functions controlling the overall calculation of one
topology optimization step
"""
import sys
import os

import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ
from NuMPI import MPI
from NuMPI.Tools import Reduction
from NuMPI.IO import save_npy

from muTopOpt.AimFunction import aim_function
from muTopOpt.AimFunction import aim_function_deriv_strain
from muTopOpt.AimFunction import aim_function_deriv_phase
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative_sequential


def calculate_dstress_dmat(cell, mat, strains, density, lambda_1, mu_1, lambda_0,
                           mu_0, order=2):
    """
    Function to calculate the partial derivative of the stress with respect
    to the material density for a polynomial interpolation of the Lame constants.
    A linear elastic, isotropic material is assumed.

    Parameters
    ----------
    cell: object
        muSpectre cell object
    mat: object
        muSpectre material object belonging to cell
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of microscopic equilibrium strains in column-major order
    density: np.ndarray([nb_quad_pts, *nb_grid_pts]) of floats between 0 and 1
        Material density for each quadrature point
    lambda_1: float
        First Lamé constant of the material at phase=1
    mu_1: float
        Second Lamé constant (shear modul) of the material at phase=1
    lambda_0: float
        First Lamé constant of the material at phase=0
    mu_0: float
        Second Lamé constant (shear modul) of the material at phase=0
    order: float
           order of the polynomial interpolation of the material. Default is 2.

    Returns
    -------
    dstress_dmat: List of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the partial derivatives of the stress with respect to the
        material density.
    """
    dim = cell.dim
    nb_grid_pts = [*cell.nb_subdomain_grid_pts]
    nb_quad_pts = cell.nb_quad_pts
    if density.size != np.prod(nb_grid_pts) * nb_quad_pts:
        message = f'The density contains only {density.size} entries, but '
        message += f'the cell contains {nb_quad_pts}x{np.prod(nb_grid_pts)}  '
        message += f'quadrature points.'
        raise ValueError(message)

    # Derivatives of Lamé constants with respect to the phase
    lame1_deriv = order * density ** (order - 1) *\
        (lambda_1 - lambda_0)
    lame2_deriv = order * density ** (order - 1) *\
        (mu_1 - mu_0)

    # Set derivatives as material parameters in mat
    for pixel_id, pixel in cell.pixels.enumerate():
        quad_id = nb_quad_pts * pixel_id
        for i in range(nb_quad_pts):
            mat.set_lame_constants(quad_id + i, lame1_deriv[(i, *tuple(pixel))],
                                   lame2_deriv[(i, *tuple(pixel))])

    # Calculate dstress_dphase
    dstress_dmat = []
    for strain in strains:
        strain = strain.reshape([dim, dim, nb_quad_pts, *nb_grid_pts], order='F')
        dstress_dmat.append(cell.evaluate_stress(strain).copy())

    # Reset material parameters of cell
    lame1 = density ** order * (lambda_1 - lambda_0) + lambda_0
    lame2 = density ** order * (mu_1 - mu_0) + mu_0
    for pixel_id, pixel in cell.pixels.enumerate():
        quad_id = nb_quad_pts * pixel_id
        for i in range(nb_quad_pts):
            mat.set_lame_constants(quad_id + i, lame1[(i, *tuple(pixel))],
                                   lame2[(i, *tuple(pixel))])

    return dstress_dmat

def sensitivity_analysis_per_quad_pts(cell, krylov_solver, strains,
                         aim_deriv_strains, aim_deriv_phase, dstress_dphase_list, tol):
    """ Worker for doing the sensitivity analysis with the adjoint method if the
        design parameters are defined per quadrature point.

    Parameters
    ----------
    cell: object
          muSpectre cell object
    krylov_solver: object
        muSpectre krylov_solver object belonging to cell
    solver_args: list
        List of additional arguments passed to the solver
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of microscopic equilibrium strains in column-major order
    aim_deriv_strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the derivativ of the aim function with respect to the strains in column-major order
    aim_deriv_phase: np.ndarray(nb_quad_pts * nb_pixels) of floats
        Derivative of the aim function with respect to the phase in column-major order
    dstress_dmat_list: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the derivative of the stresses with respect to the material density
        in column-major order
    tol: float
        Tolerance for stopping the solution of the adjoint equation

    Returns
    -------
    S: np.ndarray(nb_pixels * nb_quad_pts) of floats
       Sensitivity in column-major order
    """
    dim = cell.dim
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    # Solve the adjoint equations G:K:adjoint = -G:aim_deriv_strain
    adjoint_list = []
    for i in range(len(strains)):
        #strain = strains[i].reshape(shape, order='F')
        #cell.evaluate_stress_tangent(strain)
        rhs = aim_deriv_strains[i]
        if np.linalg.norm(rhs) > tol:
            rhs = rhs.reshape(shape, order='F')
            rhs = - cell.project(rhs).flatten(order='F')
            adjoint = krylov_solver.solve(rhs)
            adjoint = adjoint.reshape(shape, order='F')
            adjoint_list.append(adjoint.copy())
        else:
            adjoint_list.append(np.zeros(shape))

    # Sensitivity equation S = dfdrho + dKdrho:F adjoint
    S = aim_deriv_phase.flatten(order='F')
    for i in range(len(strains)):
        S += np.sum(adjoint_list[i] * dstress_dphase_list[i],
                    axis=(0, 1)).flatten(order='F')

    return S

def sensitivity_analysis_per_grid_pts(cell, krylov_solver, strains,
                         aim_deriv_strains, aim_deriv_phase, dstress_dphase_list, tol):
    """ Worker for doing the sensitivity analysis with the adjoint method if the
        design parameters (=phase) are defined at the grid points, e.g. the nodes.
        The design parameters are interpolated with linear finite elements
        (rectangular triangles) to get the material density in the elements.

    Parameters
    ----------
    cell: object
          muSpectre cell object
    krylov_solver: object
        muSpectre krylov_solver object belonging to cell
    solver_args: list
        List of additional arguments passed to the solver
    strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of microscopic equilibrium strains in column-major order
    aim_deriv_strains: list of np.ndarray(dim**2 * nb_quad_pts * nb_pixels) of floats
        List of the derivativ of the aim function with respect to the strains in column-major order
    aim_deriv_phase: np.ndarray(nb_grid_pts) of floats
        Derivative of the aim function with respect to the phase in column-major order
    dstress_dphase_list: list of np.arrays of floats
        List of the derivative of the stresses with respect to the phase in column-major order.
        Each entry has the shape: [dim, dim, nb_quad_pts, *nb_grid_pts, *nb_grid_pts]
    tol: float
        Tolerance for stopping the solution of the adjoint equation

    Returns
    -------
    S: np.ndarray(nb_pixels) of floats
       Sensitivity in column-major order
    """
    dim = cell.dim
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    # Solve the adjoint equations G:K:adjoint = -G:aim_deriv_strain
    adjoint_list = []
    for i in range(len(strains)):
        #strain = strains[i].reshape(shape, order='F')
        #cell.evaluate_stress_tangent(strain)
        rhs = aim_deriv_strains[i]
        if np.linalg.norm(rhs) > tol:
            rhs = rhs.reshape(shape, order='F')
            rhs = - cell.project(rhs).flatten(order='F')
            adjoint = krylov_solver.solve(rhs)
            adjoint = adjoint.reshape(shape, order='F')
            adjoint_list.append(adjoint.copy())
        else:
            adjoint_list.append(np.zeros(shape))

    # Sensitivity equation S = dfdrho + dKdrho:F adjoint
    S = aim_deriv_phase.flatten(order='F')
    for i in range(len(strains)):
        helper = adjoint_list[i] * dstress_dphase_list[i]
        helper = helper.reshape([-1, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts], order='F')
        helper = df_dphase_2_quad_pts_derivative_sequential(helper)
        S += np.sum(helper, axis=(0)).flatten(order='F')

    return S

def wrapper(phase, cell, mat, lambda_1, mu_1, lambda_0, mu_0, order,
                        DelFs, krylov_solver_args, solver_args, aim_args,
                        calc_sensitivity=True, folder=None):
    """ Calculate the aim function and the sensitivity for a topology
    optimization with a target stress. The optimization problem is
    regularized with a phase field approach. The discretization
    consists of linear finite elements of rectangular triangles.
    Linear elastic, isotropic materials and a polynomial
    interpolation of the Lamé constants are used as material laws.

    Parameters
    ----------
    phase: np.ndarray(nb_grid_pts) of floats
           Design parameters. Corresponds to the material
           density at each quadrature point.
    cell: object
          muSpectre cell object
    mat: object
         muSpectre cell object belonging to cell
    lambda_1: float
        First Lamé constant of the material at phase=1
    mu_1: float
        Second Lamé constant (shear modul) of the material at phase=1
    lambda_0: float
        First Lamé constant of the material at phase=0
    mu_0: float
        Second Lamé constant (shear modul) of the material at phase=0
    order: float
           Polynomial order of the material interpolation
    DelFs: list of np.ndarray(dim, dim) of floats
        List of prescribed macroscopic strain
    krylov_solver_args: list
        List of additional arguments passed to the krylov_solver
    solver_args: list
        List of additional arguments passed to the solver
    aim_args: list
        list with additional arguments passed to the aim function
    calc_sensitivity: boolean
         If False, the sensitivity is not calculated. Default is True.
    folder: string or None
            Name of a folder in which to save the results. If None, the
            results are not saved. If a string, the following files are saved:
           * file_tmp.npy: for saving the new phase before the muSpectre
                calculation. After the muSpectre calculation, this file is deleted.
           * file_last.npy: for saving the phase of the last
                 successful optimization step
           * file_evo.txt: save the aim function in each optimization step

    Returns
    -------
    aim: float
         Aim function
    S: np.ndarray(nb_pixels * nb_quad_pts) of floats
       Sensitivity in Column-major order
    """
    phase = phase.reshape(cell.nb_subdomain_grid_pts, order='F')
    # Save temporary phase (mainly for debugging)
    if folder is not None:
        name = folder + f'phase_tmp.npy'
        save_npy(name, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

    # Change material of cell
    density = node_to_quad_pt_2_quad_pts_sequential(phase)
    density = density.flatten(order='F')
    lame1 = (lambda_1 - lambda_0) * density ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density ** order + mu_0
    for pixel_id, pixel in cell.pixels.enumerate():
        for i in range(cell.nb_quad_pts):
            quad_id = cell.nb_quad_pts * pixel_id + i
            mat.set_lame_constants(quad_id, lame1[quad_id], lame2[quad_id])

    # Solve the equilibrium equations
    dim = cell.dim
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

    # TODO: Give average stresses to functions to avoid recalculation
    average_stresses = []
    for stress in stresses:
        stress_average = np.empty((dim, dim))
        for i in range(dim):
            for j in range(dim):
                stress_average[i, j] = Reduction(MPI.COMM_WORLD).mean(stress[i, j])
        average_stresses.append(stress_average)

    # Calculate the aim function
    aim = aim_function(cell, phase, strains, stresses, *aim_args)

    if calc_sensitivity:
        # Partial derivatives
        density = density.reshape([cell.nb_quad_pts,
                               *cell.nb_subdomain_grid_pts], order='F')
        dstress_dmat_list =\
            calculate_dstress_dmat(cell, mat, strains, density,
                                     lambda_1, mu_1, lambda_0,
                                     mu_0, order=order)
        aim_deriv_strains =\
            aim_function_deriv_strain(cell, strains, stresses, *aim_args)
        aim_deriv_phase =\
            aim_function_deriv_phase(cell, phase, strains, stresses,
                                     dstress_dmat_list, *aim_args)

        S = sensitivity_analysis_per_grid_pts(cell, krylov_solver, strains,
                                 aim_deriv_strains, aim_deriv_phase,
                                 dstress_dmat_list, solver_args[1])

    # Remove file with temporary phase if muSpectre calculations worked
    if (MPI.COMM_WORLD.rank == 0) and (folder is not None):
        name = folder + f'phase_tmp.npy'
        if name is not None:
            os.remove(name)

    # Save the last step
    if folder is not None:
        phase = phase.reshape(cell.nb_subdomain_grid_pts, order='F')
        name = folder + f'phase_last.npy'
        save_npy(name, phase, tuple(cell.subdomain_locations),
                 tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
        if (MPI.COMM_WORLD.rank == 0):
            name = folder + 'evolution.txt'
            with open(name, 'a') as f:
                if calc_sensitivity:
                    norm_S = np.linalg.norm(S)**2
                    norm_S = np.sqrt(Reduction(MPI.COMM_WORLD).sum(norm_S))
                    np.savetxt(f, [aim, norm_S], delimiter=' ', newline=' ')
                else:
                    np.savetxt(f, [aim], delimiter=' ', newline=' ')
                for average_stress in average_stresses:
                    np.savetxt(f, average_stress.flatten(),
                               delimiter=' ', newline=' ')
                print('', file=f)

    if calc_sensitivity:
        return aim, S.flatten(order='F')
    else:
        return aim

def wrapper_at_quad_pts(phase, cell, mat, lambda_1, mu_1, lambda_0, mu_0, order,
                        DelFs, krylov_solver_args, solver_args, aim_args,
                        calc_sensitivity=True, folder=None):
    """ Calculate the aim function and the sensitivity for a topology
    optimization with a target stress. The optimization problem is
    regularized with a phase field approach. The discretization
    consists of linear finite elements of rectangular triangles.
    Linear elastic, isotropic materials and a polynomial
    interpolation of the Lamé constants are used as material laws.

    Parameters
    ----------
    phase: np.ndarray(nb_quad_pts, *nb_grid_pts) of floats
           Design parameters. Corresponds to the material
           density at each quadrature point.
    cell: object
          muSpectre cell object
    mat: object
         muSpectre cell object belonging to cell
    lambda_1: float
        First Lamé constant of the material at phase=1
    mu_1: float
        Second Lamé constant (shear modul) of the material at phase=1
    lambda_0: float
        First Lamé constant of the material at phase=0
    mu_0: float
        Second Lamé constant (shear modul) of the material at phase=0
    order: float
           Polynomial order of the material interpolation
    DelFs: list of np.ndarray(dim, dim) of floats
        List of prescribed macroscopic strain
    krylov_solver_args: list
        List of additional arguments passed to the krylov_solver
    solver_args: list
        List of additional arguments passed to the solver
    aim_args: list
        list with additional arguments passed to the aim function
    calc_sensitivity: boolean
         If False, the sensitivity is not calculated. Default is True.
    folder: string or None
            Name of a folder in which to save the results. If None, the
            results are not saved. If a string, the following files are saved:
           * file_tmp.npy: for saving the new phase before the muSpectre
                calculation. After the muSpectre calculation, this file is deleted.
           * file_last.npy: for saving the phase of the last
                 successful optimization step
           * file_evo.txt: save the aim function in each optimization step

    Returns
    -------
    aim: float
         Aim function
    S: np.ndarray(nb_pixels * nb_quad_pts) of floats
       Sensitivity in Column-major order
    """
    # Save temporary phase (mainly for debugging)
    if folder is not None:
        for i in range(cell.nb_quad_pts):
            phase = phase.reshape([cell.nb_quad_pts, *cell.nb_subdomain_grid_pts],
                                  order='F')
            name = folder + f'phase_tmp_quad_pt_[i].npy'
            save_npy(name, phase[i], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

    # Change material of cell
    density = phase
    density = density.flatten(order='F')
    lame1 = (lambda_1 - lambda_0) * density ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density ** order + mu_0
    for pixel_id, pixel in cell.pixels.enumerate():
        for i in range(cell.nb_quad_pts):
            quad_id = cell.nb_quad_pts * pixel_id + i
            mat.set_lame_constants(quad_id, lame1[quad_id], lame2[quad_id])

    # Solve the equilibrium equations
    dim = cell.dim
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

    # TODO: Give average stresses to functions to avoid recalculation
    average_stresses = []
    for stress in stresses:
        stress_average = np.empty((dim, dim))
        for i in range(dim):
            for j in range(dim):
                stress_average[i, j] = Reduction(MPI.COMM_WORLD).mean(stress[i, j])
        average_stresses.append(stress_average)

    # Calculate the aim function
    aim = aim_function(cell, phase, strains, stresses, *aim_args)

    if calc_sensitivity:
        # Partial derivatives
        phase = phase.reshape([cell.nb_quad_pts,
                               *cell.nb_subdomain_grid_pts], order='F')
        dstress_dphase_list =\
            calculate_dstress_dphase(cell, mat, strains, phase,
                                     lambda_1, mu_1, lambda_0,
                                     mu_0, order=order)
        aim_deriv_strains =\
            aim_function_deriv_strain(cell, strains, stresses, *aim_args)
        aim_deriv_phase =\
            aim_function_deriv_phase(cell, phase, strains, stresses,
                                     dstress_dphase_list, *aim_args)

        S = sensitivity_analysis_per_quad_pts(cell, krylov_solver, strains,
                                 aim_deriv_strains, aim_deriv_phase,
                                 dstress_dphase_list, solver_args[1])

    # Remove file with temporary phase if muSpectre calculations worked
    if (MPI.COMM_WORLD.rank == 0) and (folder is not None):
        for i in range(cell.nb_quad_pts):
            name = folder + f'phase_tmp_quad_pt_{i}.npy'
            if name is not None:
                os.remove(name)

    # Save the last step
    if folder is not None:
        phase = phase.reshape([cell.nb_quad_pts, *cell.nb_subdomain_grid_pts],
                              order='F')
        for i in range(cell.nb_quad_pts):
            name = folder + f'phase_last_quad_pt_{i}.npy'
            save_npy(name, phase[i], tuple(cell.subdomain_locations),
                     tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
        if (MPI.COMM_WORLD.rank == 0):
            name = folder + 'evolution.txt'
            with open(name, 'a') as f:
                if calc_sensitivity:
                    norm_S = np.linalg.norm(S)**2
                    norm_S = np.sqrt(Reduction(MPI.COMM_WORLD).sum(norm_S))
                    np.savetxt(f, [aim, norm_S], delimiter=' ', newline=' ')
                else:
                    np.savetxt(f, [aim], delimiter=' ', newline=' ')
                for average_stress in average_stresses:
                    np.savetxt(f, average_stress.flatten(),
                               delimiter=' ', newline=' ')
                print('', file=f)

    if calc_sensitivity:
        return aim, S.flatten(order='F')
    else:
        return aim
