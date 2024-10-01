"""
Test different functions wether they work in parallel.
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

import muTopOpt.AimFunction as aim
import muTopOpt.MaterialDensity as dens
import muTopOpt.PhaseField as phf
import muTopOpt.Controller as contr

def test_aim_function_parallel():
    """ Test the aim function and its derivatives for parallel calculations.
    """
    ### ----- Set-up ----- ###
    if MPI.COMM_WORLD.rank == 0:
        print(f'MPI.COMM_WORLD.size = {MPI.COMM_WORLD.size}')
    # Discretization
    nb_grid_pts = [5, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    fft = 'mpi'

    # Material
    rng = np.random.default_rng(1231414)
    phase = rng.random(nb_grid_pts)
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

    # Target stresses
    lame1_av = (lambda_1 - lambda_0) / 2
    lame2_av = (mu_1 - mu_0) / 2
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * lame2_av * DelF + lame1_av * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)

    ### ----- Parallel calculation ----- ###
    # Cell initialisation
    cell_paral = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights, fft=fft, communicator=MPI.COMM_WORLD)
    phase_paral = phase[cell_paral.fft_engine.subdomain_slices]

    # Material initialization
    density = dens.node_to_quad_pt_2_quad_pts(phase_paral, cell_paral)
    density = density.reshape([nb_quad_pts, -1], order='F')
    lame1 = (lambda_1 - lambda_0) * density ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density ** order + mu_0
    mat = µ.material.MaterialElasticLocalLame_2d.make(cell_paral, "material")
    for pixel_id, pixel in cell_paral.pixels.enumerate():
        mat.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell_paral.initialise()

    # muSpectre calculation
    solver = µ.solvers.KrylovSolverCG(cell_paral, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell_paral, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    # Aim function
    aim_paral = aim.aim_function(cell_paral, phase_paral, strains, stresses,
                                 target_stresses, eta, weight_phase_field)
    # Derivatives
    aim_deriv_strain_paral =\
        aim.aim_function_deriv_strain(cell_paral, strains, stresses,
                                      target_stresses, eta,
                                      weight_phase_field)
    density = density.reshape([nb_quad_pts,
                               *cell_paral.nb_subdomain_grid_pts], order='F')
    dstress_dmat_list = contr.calculate_dstress_dmat(cell_paral, mat, strains,
                                                     density, lambda_1,
                                                     mu_1, lambda_0, mu_0,
                                                     order=order)
    aim_deriv_phase_paral =\
        aim.aim_function_deriv_phase(cell_paral, phase_paral, strains, stresses,
                                     dstress_dmat_list, target_stresses,
                                     eta, weight_phase_field)

    ### ----- Sequential calculation ----- ###
    cell_seq = µ.Cell(nb_grid_pts, lengths, formulation,
                      gradient, weights)
    # Material initialization
    density_seq = dens.node_to_quad_pt_2_quad_pts_sequential(phase)
    density_seq = density_seq.reshape([nb_quad_pts, -1], order='F')
    lame1 = (lambda_1 - lambda_0) * density_seq ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density_seq ** order + mu_0
    mat_seq = µ.material.MaterialElasticLocalLame_2d.make(cell_seq, "material")
    for pixel_id, pixel in cell_seq.pixels.enumerate():
        mat_seq.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell_seq.initialise()

    # muSpectre calculation
    solver=µ.solvers.KrylovSolverCG(cell_seq, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell_seq, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy()
        stresses.append(stress)
        strain = r.grad.copy()
        strains.append(strain)

    # Aim function
    aim_seq = aim.aim_function_sequential(cell_seq, phase, strains, stresses,
                                          target_stresses, eta, weight_phase_field)

    if MPI.COMM_WORLD.rank == 0:
        print(f'Aim function (parallel)   = {aim_paral}')
        print(f'Aim function (sequential) = {aim_seq}')
        assert abs(aim_paral - aim_seq) < 1e-7
        print()

    # Derivative with respect to the strains
    aim_deriv_strain_seq =\
        aim.aim_function_deriv_strain_sequential(cell_seq, strains, stresses,
                                                 target_stresses, eta, weight_phase_field)

    for i in range(len(DelFs)):
        helper = aim_deriv_strain_paral[i]
        helper = helper.reshape((dim, dim, nb_quad_pts, *
                                   cell_paral.nb_subdomain_grid_pts), order='F')
        helper2 = aim_deriv_strain_seq[i]
        helper2 = helper2.reshape((dim, dim, nb_quad_pts, *nb_grid_pts), order='F')
        helper3 = np.empty(helper.shape)
        for j in range(dim):
            for k in range(dim):
                for l in range(nb_quad_pts):
                    helper3[j, k, l] = helper2[(j, k, l,
                                                *cell_paral.fft_engine.subdomain_slices)]
        err = np.linalg.norm(helper - helper3)
        assert err < 1e-7

    # Derivative with respect to the phase
    density_seq = density_seq.reshape([nb_quad_pts,
                               *cell_seq.nb_subdomain_grid_pts], order='F')
    dstress_dmat_list = contr.calculate_dstress_dmat(cell_seq, mat_seq, strains,
                                                     density_seq, lambda_1,
                                                     mu_1, lambda_0, mu_0,
                                                     order=order)
    aim_deriv_phase_seq =\
        aim.aim_function_deriv_phase_sequential(cell_seq, phase, strains, stresses,
                                                dstress_dmat_list, target_stresses,
                                                eta, weight_phase_field)
    err = aim_deriv_phase_seq[cell_paral.fft_engine.subdomain_slices] - aim_deriv_phase_paral
    assert np.linalg.norm(err) < 1e-7

    if MPI.COMM_WORLD.rank == 0:
        print('Finished test_aim_function_parallel.')
        print()

def test_material_density_parallel(verbose=True):
    ### ----- Set-up ----- ###
    if verbose and MPI.COMM_WORLD.rank == 0:
        print(f'MPI.COMM_WORLD.size = {MPI.COMM_WORLD.size}')
    # Discretization
    nb_grid_pts = [3, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    fft = 'mpi'

    rng = np.random.default_rng(1231414)
    phase = rng.random(nb_grid_pts)
    df_dmat = rng.random((4, nb_quad_pts, *nb_grid_pts))
    df_dmat[0, 0, 1, 2] = -0.5

    ### ----- Parallel material density ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights, fft=fft, communicator=MPI.COMM_WORLD)
    phase_paral = phase[cell.fft_engine.subdomain_slices].copy()
    density_paral = dens.node_to_quad_pt_2_quad_pts(phase_paral, cell)
    norm_paral = np.linalg.norm(density_paral) ** 2
    norm_paral = np.sqrt(Reduction(MPI.COMM_WORLD).sum(norm_paral))

    df_dmat_paral = np.empty((4, nb_quad_pts, *cell.nb_subdomain_grid_pts))
    for i in range(4):
        for j in range(nb_quad_pts):
            df_dmat_paral[i, j] = df_dmat[(i, j, *cell.fft_engine.subdomain_slices)]
    df_dphase_paral = dens.df_dphase_2_quad_pts_derivative(df_dmat_paral, cell)

    ### ----- Sequential material density ----- ###
    density_seq = dens.node_to_quad_pt_2_quad_pts_sequential(phase)
    norm_seq = np.linalg.norm(density_seq)

    df_dphase_seq = dens.df_dphase_2_quad_pts_derivative_sequential(df_dmat)

    ### ----- Comparison ----- ###
    # Density
    for i in range(cell.nb_quad_pts):
        err = density_seq[(i, *cell.fft_engine.subdomain_slices)] - density_paral[i]
        err = np.linalg.norm(err)
        if verbose:
            message = f'Rank {MPI.COMM_WORLD.rank}: Difference between densities '
            message += f' at quad pt {i} = {err}'
            print(message)
        assert err < 1e-7
        assert abs(norm_seq - norm_paral) < 1e-7

    # Derivative of density
    for i in range(2):
        helper = df_dphase_seq[(i, *cell.fft_engine.subdomain_slices)]
        err = helper - df_dphase_paral[i]
        assert np.linalg.norm(err) < 1e-7

    if MPI.COMM_WORLD.rank == 0:
        print('Finished test_material_density_parallel.')
        print()

def test_phase_field_parallel(verbose=True):
    ### ----- Set-up ----- ###
    if verbose and MPI.COMM_WORLD.rank == 0:
        print(f'MPI.COMM_WORLD.size = {MPI.COMM_WORLD.size}')
    # Discretization
    nb_grid_pts = [3, 3]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)
    eta = 0.2

    fft = 'mpi'

    rng = np.random.default_rng(1231414)
    phase = rng.random(nb_grid_pts)

    ### ----- Parallel phase field terms ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights, fft=fft, communicator=MPI.COMM_WORLD)
    phase_paral = phase[cell.fft_engine.subdomain_slices]

    grad_paral = phf.phase_field_gradient(phase_paral, cell)
    pot_paral = phf.phase_field_double_well_potential(phase_paral, cell)
    comp_paral = phf.phase_field_energy(phase_paral, cell, eta)

    ### ----- Sequential phase field terms ----- ###
    cell_seq = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                      weights)
    grad_seq = phf.phase_field_gradient_sequential(phase, cell_seq)
    pot_seq = phf.phase_field_double_well_potential_sequential(phase, cell_seq)
    comp_seq = phf.phase_field_energy_sequential(phase, cell_seq, eta)

    if MPI.COMM_WORLD.rank == 0:
        if verbose:
            print(f'Grad_term (seq)   = {grad_seq}')
            print(f'Grad_term (paral) = {grad_paral}')
            print(f'pot_term (seq)   = {pot_seq}')
            print(f'pot_term (paral) = {pot_paral}')
            print(f'complete (seq)   = {comp_seq}')
            print(f'complete (paral) = {comp_paral}')
        assert abs(grad_seq - grad_paral) < 1e-7
        assert abs(pot_seq - pot_paral) < 1e-7
        assert abs(comp_seq - comp_paral) < 1e-7
        print('Finished test_phase_field_parallel.')
        print()

def test_wrapper_parallel():
    ### ----- Set-up ----- ###
    if MPI.COMM_WORLD.rank == 0:
        print(f'MPI.COMM_WORLD.size = {MPI.COMM_WORLD.size}')
    # Discretization
    nb_grid_pts = [3, 5]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    gradient, weights = µ.linear_finite_elements.gradient_2d
    nb_quad_pts = len(weights)
    nb_pixels = np.prod(nb_grid_pts)

    fft = 'mpi'

    # Material
    rng = np.random.default_rng(1231414)
    phase = rng.random(nb_grid_pts)
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
    #DelFs = [np.zeros([dim, dim])]
    loading = 0.013
    DelFs[0][0, 0] = loading
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (tol, maxiter, verbose)
    solver_args = (tol, tol, verbose)

    # Target stresses
    lame1_av = (lambda_1 - lambda_0) / 2
    lame2_av = (mu_1 - mu_0) / 2
    target_stresses = []
    for DelF in DelFs:
        stress = 2 * lame2_av * DelF + lame1_av * np.trace(DelF) * np.eye(dim)
        target_stresses.append(stress)
    aim_args = (target_stresses, eta, weight_phase_field)

    ### ----- Parallel calculation ----- ###
    # Cell initialisation
    cell_paral = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
                  weights, fft=fft, communicator=MPI.COMM_WORLD)
    phase_paral = phase[cell_paral.fft_engine.subdomain_slices]

    # Material initialization
    density_paral = dens.node_to_quad_pt_2_quad_pts(phase_paral, cell_paral)
    density_paral = density_paral.reshape([nb_quad_pts, -1], order='F')
    lame1 = (lambda_1 - lambda_0) * density_paral ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density_paral ** order + mu_0
    mat_paral = µ.material.MaterialElasticLocalLame_2d.make(cell_paral, "material")
    for pixel_id, pixel in cell_paral.pixels.enumerate():
        mat_paral.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell_paral.initialise()

    # Calculate aim function and sensitivity
    aim_paral, S_paral =\
        contr.wrapper(phase_paral, cell_paral, mat_paral, lambda_1, mu_1,
                      lambda_0, mu_0, order, DelFs, krylov_solver_args,
                      solver_args, aim_args, calc_sensitivity=True, folder=None)

    ### ----- Sequential calculation ----- ###
    cell_seq = µ.Cell(nb_grid_pts, lengths, formulation,
                      gradient, weights)
    # Material initialization
    density_seq = dens.node_to_quad_pt_2_quad_pts_sequential(phase)
    density_seq = density_seq.reshape([nb_quad_pts, -1], order='F')
    lame1 = (lambda_1 - lambda_0) * density_seq ** order + lambda_0
    lame2 = (mu_1 - mu_0) * density_seq ** order + mu_0
    mat_seq = µ.material.MaterialElasticLocalLame_2d.make(cell_seq, "material")
    for pixel_id, pixel in cell_seq.pixels.enumerate():
        mat_seq.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
    cell_seq.initialise()

    # Calculate aim function and sensitivity
    aim_seq, S_seq =\
        contr.wrapper_sequential(phase, cell_seq, mat_seq, lambda_1, mu_1,
                                 lambda_0, mu_0, order, DelFs, krylov_solver_args,
                                 solver_args, aim_args, calc_sensitivity=True,
                                 folder=None)

    # Compare aim function
    if MPI.COMM_WORLD.rank == 0:
        print(f'Aim function (paral) = {aim_paral}')
        print(f'Aim function (seq)   = {aim_seq}')
    assert abs(aim_seq - aim_paral) < 1e-7

    # Compare sensitivity
    S_paral = S_paral.reshape(cell_paral.nb_subdomain_grid_pts, order='F')
    S_seq = S_seq.reshape(nb_grid_pts, order='F')
    err = S_seq[cell_paral.fft_engine.subdomain_slices] - S_paral
    assert np.linalg.norm(err) < 1e-7

    if MPI.COMM_WORLD.rank == 0:
        print('Finished test_wrapper_parallel.')
        print()

if __name__ == "__main__":
    test_material_density_parallel()
    # test_phase_field_parallel()
    # test_aim_function_parallel()
    test_wrapper_parallel()
