"""
@file   optimize_negative_poisson.py

@author Indre Jödicke <indre.joedicke@imtek.uni-freiburg.de>

@date   26 Jun 2023

@brief  Example to optimize for a negative Poissons ratio.
"""

import os
import shutil

import muSpectre as µ
import muFFT
import muSpectre.sensitivity_analysis as sa

import numpy as np
import time
from NuMPI.Optimization import LBFGS
from NuMPI import MPI
from NuMPI.IO import save_npy
from NuMPI.IO import load_npy
from NuMPI.Tools import Reduction

from muTopOpt.Controller import call_function
from muTopOpt.Filter import map_to_unit_range

################################################################################
### ------------------------ Parameter declarations ------------------------ ###
################################################################################
t = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f'MPI: Rank {MPI.COMM_WORLD.rank}, size {MPI.COMM_WORLD.size}')

### ----- Geometry ----- ###
nb_grid_pts = [8, 8]
lengths = [1, 1]

dim = len(nb_grid_pts)
hx = lengths[0] / nb_grid_pts[0]
hy = lengths[1] / nb_grid_pts[1]

### ----- Target elastic constants ----- ###
mu_given = 25
lambda_given = -10

### ----- Weighting paramaters ----- ###
w = 1 #0.0014 #0.003
eta = 0.025

### ----- Formulation ----- ###
formulation = µ.Formulation.small_strain

### ----- FFT-engine ----- ###
fft = 'mpi' # Seriel

### ----- Random seed ----- ###
# For random phase distribution
np.random.seed(1130)

### ----- Macroscopic strain ----- ###
loading = 0.01
DelFs = [np.zeros([dim, dim])]
DelFs[0][1, 1] = loading

### ----- Material parameters ----- ###
Young1 = 0
Young2 = 100
Poisson1 = 0
Poisson2 = 0

### ----- muSpectre solver parameters ----- ###
newton_tol       = 1e-4
cg_tol           = 1e-5 # tolerance for cg algo
equil_tol        = 1e-4 # tolerance for equilibrium
maxiter          = 15000
verbose          = µ.Verbosity.Silent
krylov_solver_args = (cg_tol, maxiter, verbose)
solver_args = (newton_tol, equil_tol, verbose)

nb_strain_steps = 1

### ----- Numerical derivatives ----- ###
# Use linear finite elements (rectangular triangles)
gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
            muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
weights = [1, 1]

### ----- Optimization parameters ----- ###
gtol = 1e-5
ftol = 1e-15
maxcor = 10

### ----- Folder for saving data ----- ###
# Several files with results are saved in folder:
# evolution_of_aim_function.txt: All values of aim function
# evolution_of_aim_function_details.txt: Contributions to the aim function + norm of sensitivity
# metadata.txt: Metadata for runing the optimization
# metadata_convergence.txt: Metadata for the convergence of the optimizer
# phase_final.npy: Final phase of the finished optimization
# phase_ini.npy: Initial phase
# phase_last_step.npy: Last successful optimization step

folder = f'results_negative_poisson'

################################################################################
### ----------------------- Aimed for average stress ----------------------- ###
################################################################################
### ----- Aimed for elastic constants ----- ###
Young_aim = mu_given * (3*lambda_given + 2*mu_given) / (lambda_given + mu_given)
Poisson_aim = 0.5 * lambda_given / (lambda_given + mu_given)

### ----- Construct cell with aimed for material properties ----- ###
cell = µ.Cell([1, 1], lengths, formulation, fft=fft)
mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
mat.add_pixel(0, Young_aim, Poisson_aim)

### ----- Calculate the aimed for average stresses ----- ###
nb_pixels = cell.nb_pixels
target_stresses = []
for DelF in DelFs:
    strain = DelF
    strain = strain.reshape([*DelF.shape, 1, 1], order='F')
    target_stress = cell.evaluate_stress(strain)
    target_stress = np.average(target_stress, axis=(2, 3, 4))
    # Nondimensionalization
    target_stress = target_stress / Young2 / loading
    target_stresses.append(target_stress)

# Arguments passed to the aim function
args = (target_stresses, w, eta, loading, Young2)

################################################################################
### ---------------------- Save important information ---------------------- ###
################################################################################
### ----- Create folder for saving data ----- ###
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        shutil.rmtree(folder)
        os.makedirs(folder)

### ----- Print interesting data ----- ###
if MPI.COMM_WORLD.rank == 0:
    print(µ.version.info())

    print('Optimize')
    print('  Initial phase: Random')
    print('  Grid: Rectangular triangles')
    print('  mu_target =', mu_given)
    print('  lambda_target =', lambda_given)
    print('  nb_grid_pts =', nb_grid_pts)

### ----- Save metadata of simulation ----- ###
file_name = folder + '/metadata.txt'
if MPI.COMM_WORLD.rank == 0:
    with open(file_name, 'w') as f:
        ### ----- Save metadata ----- ###
        print('nb_grid_pts, lenghts, weight, '
              'weight_phase_field, Young1, Poisson1,  Young2, Poisson2, '
              'newton_tolerance, cg_tolerance, equil_tolerance, maxiter, '
              'gtol, ftol, '
              'maxcor, loading, target_shear_modulus, target_lame2, '
              'nb_strain_steps', file=f)
        np.savetxt(f, nb_grid_pts, delimiter=' ', newline=' ')
        np.savetxt(f, lengths, delimiter=' ', newline=' ')
        np.savetxt(f, [w], delimiter=' ', newline=' ')
        np.savetxt(f, [eta], delimiter=' ', newline=' ')
        np.savetxt(f, [Young1], delimiter=' ', newline=' ')
        np.savetxt(f, [Poisson1], delimiter=' ', newline=' ')
        np.savetxt(f, [Young2], delimiter=' ', newline=' ')
        np.savetxt(f, [Poisson2], delimiter=' ', newline=' ')
        np.savetxt(f, [newton_tol], delimiter=' ', newline=' ')
        np.savetxt(f, [cg_tol], delimiter=' ', newline=' ')
        np.savetxt(f, [equil_tol], delimiter=' ', newline=' ')
        np.savetxt(f, [maxiter], delimiter=' ', newline=' ')
        np.savetxt(f, [gtol], delimiter=' ', newline=' ')
        np.savetxt(f, [ftol], delimiter=' ', newline=' ')
        np.savetxt(f, [maxcor], delimiter=' ', newline=' ')
        np.savetxt(f, [loading], delimiter=' ', newline=' ')
        np.savetxt(f, [mu_given], delimiter=' ', newline=' ')
        np.savetxt(f, [lambda_given], newline=' ')
        np.savetxt(f, [nb_strain_steps], delimiter=' ')

        print('MPI_size', file=f)
        print(MPI.COMM_WORLD.size, file=f)

### ----- Save initial phase distribution ----- ###
# Cell construction
cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights=weights,
              fft=fft, communicator=MPI.COMM_WORLD)

# Initial phase distribution
phase_ini = np.random.random_sample(cell.nb_domain_grid_pts)
phase_ini = phase_ini[cell.fft_engine.subdomain_slices]

# Saving
file_name = folder + '/phase_ini.npy'
save_npy(file_name, phase_ini, tuple(cell.subdomain_locations),
         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
phase_ini = phase_ini.flatten(order='F')
phase = phase_ini.copy()

################################################################################
### ----------------------------- Optimization ----------------------------- ###
################################################################################
### ----- Initialisation ----- ###
# Material
Young = (Young2 - Young1) * map_to_unit_range(phase) + Young1
Poisson = (Poisson2 - Poisson1) * map_to_unit_range(phase) + Poisson1
mat2 = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
for pixel_id, pixel in cell.pixels.enumerate():
    mat2.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])
cell.initialise()

# Passing function arguments in NuMPI conform way
file_tmp = folder + '/phase_tmp.npy'
file_last = folder + '/phase_last_step.npy'
file_evo = folder + '/evolution_of_aim_function.txt'
file_evo_details = folder + '/evolution_of_aim_function_details.txt'
opt_args = (cell, mat2, Young1, Poisson1, Young2, Poisson2, DelFs,
            nb_strain_steps, krylov_solver_args, solver_args, args, gradient, weights, True,
            file_tmp, file_last, file_evo, file_evo_details)
if MPI.COMM_WORLD.rank == 0:
    with open(file_evo_details, 'w') as f:
        title = 'aim_function  error_target_stresses  weighted_phase_field  norm_sensitivity'
        print(title, file=f)

def fun(x):
    return call_function(x, *opt_args)


### ----- Optimization ----- ###
t = time.time()
# With NuMPI optimizer
opt_result = LBFGS(fun, phase, args=opt_args, jac=True, gtol=gtol,
                   ftol=ftol, maxcor=maxcor, comm=MPI.COMM_WORLD, max_ls=100)

################################################################################
### ----------------------------- Save results ----------------------------- ###
################################################################################
### ----- Save metadata of convergence ----- ###
if MPI.COMM_WORLD.rank == 0:
    file_name = folder + '/metadata_convergence.txt'
    with open(file_name, 'a') as f:
        print('opt_success, opt_time (min), opt_nb_of_iteration', file=f)
        np.savetxt(f, [opt_result.success, (time.time() - t) / 60, opt_result.nit],
                   delimiter=' ', newline=' ')
        if not opt_result.success:
            print(file=f)
            print(opt_result.message, file=f)

### ----- Save final phase distribution ----- ###
# Final phase
phase_final = opt_result.x.reshape(*cell.nb_subdomain_grid_pts, order='F')
file_name = folder + '/phase_final.npy'
save_npy(file_name, phase_final, tuple(cell.subdomain_locations),
         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

t = time.time() - t
if MPI.COMM_WORLD.rank == 0:
    print()
    if opt_result.success:
        print('Optimization successful')
    else:
        print('Optimization failed')
    print('Time (min) =', t/60)
