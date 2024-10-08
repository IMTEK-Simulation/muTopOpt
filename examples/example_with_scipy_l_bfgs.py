"""
@file   example_with_scipy_l_bfgs.py

@author Indre Jödicke <indre.joedicke@imtek.uni-freiburg.de>

@date   19 Sep 2024

@brief  Example for a topology optimization using
        the scipy L-BFGS-B optimizer.
"""
import sys
import os
import shutil

import numpy as np
import scipy.optimize as opt
import time

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ
import muFFT

from muTopOpt.Controller import wrapper

from NuMPI import MPI
from NuMPI.IO import save_npy


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

gradient, weights = µ.linear_finite_elements.gradient_2d

### ----- Target elastic constants ----- ###
poisson_target = -0.5
young_target = 0.25

mu_target = young_target / 2 / (1 + poisson_target)
lambda_target = young_target * poisson_target
lambda_target = lambda_target / (1 + poisson_target) / (1 - 2 * poisson_target)

### ----- Weighting parameters ----- ###
eta = 0.01
weight_phase_field = 0.01

### ----- Formulation ----- ###
formulation = µ.Formulation.small_strain

### ----- FFT-engine ----- ###
fft = 'mpi' #'fftwmpi' # Parallel
# fft = 'fftw' # Seriel

### ----- Random seed ----- ###
# For random phase distribution
np.random.seed(1130)

### ----- Macroscopic strain ----- ###
DelFs = [np.zeros([dim, dim])]
DelFs[0][1, 1] = 1.1
DelFs[0][0, 0] = 1.

### ----- Material parameter ----- ###
order = 2 # Interpolation order

# First material: Solid
young = 1
poisson = 0.2
mu_1 = young / 2 / (1 + poisson)
lambda_1 = young * poisson / (1 + poisson) / (1 - 2 * poisson)

# Second material: Void
mu_0 = 0
lambda_0 = 0


### ----- muSpectre solver parameters ----- ###
newton_tol       = 1e-4
cg_tol           = 1e-5 # tolerance for cg algo
equil_tol        = 1e-4 # tolerance for equilibrium
maxiter          = 15000
verbose          = µ.Verbosity.Silent
krylov_solver_args = (cg_tol, maxiter, verbose)
solver_args = (newton_tol, equil_tol, verbose)


### ----- Optimization parameters ----- ###
gtol = 1e-5
ftol = 1e-15
maxcor = 10

### ----- Folder for saving data ----- ###
folder = f'examples/results_scipy/'

################################################################################
### ----------------------------- Target stress ---------------------------- ###
################################################################################
### ----- Construct cell with aimed for material properties ----- ###
cell = µ.Cell([1, 1], lengths, formulation, fft=fft)
mat = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
mat.add_pixel_lame(0, lambda_target, mu_target)

### ----- Calculate the aimed for average stresses ----- ###
target_stresses = []
for DelF in DelFs:
    strain = DelF
    strain = strain.reshape([*DelF.shape, 1, 1], order='F')
    target_stress = cell.evaluate_stress(strain)
    target_stress = np.average(target_stress, axis=(2, 3, 4))
    target_stresses.append(target_stress)

# Arguments passed to the aim function
aim_args = (target_stresses, eta, weight_phase_field)


################################################################################
### ---------------------- Save important information ---------------------- ###
################################################################################
### ----- Create folder for saving data ----- ###
if not os.path.exists(folder):
    os.makedirs(folder)
else:
    shutil.rmtree(folder)
    os.makedirs(folder)

### ----- Print some data ----- ###
print(µ.version.info())

print('Optimize')
print('  mu_target =', mu_target)
print('  lambda_target =', lambda_target)
print('  nb_grid_pts =', nb_grid_pts)

### ----- Save metadata of simulation ----- ###
file_name = folder + '/metadata.txt'
with open(file_name, 'w') as f:
    ### ----- Save metadata ----- ###
    print('nb_grid_pts, lenghts, weight_eta, weight_phase_field, first_lame_1, '
          'second_lame_1, first_lame_0, second_lame_0, '
          'newton_tolerance, cg_tolerance, equil_tolerance, maxiter, '
          'gtol, ftol, maxcor, first_lame_target, second_lame_target, ',
          'loading', file=f)
    np.savetxt(f, nb_grid_pts, delimiter=' ', newline=' ')
    np.savetxt(f, lengths, delimiter=' ', newline=' ')
    np.savetxt(f, [eta], delimiter=' ', newline=' ')
    np.savetxt(f, [weight_phase_field], delimiter=' ', newline=' ')
    np.savetxt(f, [lambda_1], delimiter=' ', newline=' ')
    np.savetxt(f, [mu_1], delimiter=' ', newline=' ')
    np.savetxt(f, [lambda_0], delimiter=' ', newline=' ')
    np.savetxt(f, [mu_0], delimiter=' ', newline=' ')
    np.savetxt(f, [newton_tol], delimiter=' ', newline=' ')
    np.savetxt(f, [cg_tol], delimiter=' ', newline=' ')
    np.savetxt(f, [equil_tol], delimiter=' ', newline=' ')
    np.savetxt(f, [maxiter], delimiter=' ', newline=' ')
    np.savetxt(f, [gtol], delimiter=' ', newline=' ')
    np.savetxt(f, [ftol], delimiter=' ', newline=' ')
    np.savetxt(f, [maxcor], delimiter=' ', newline=' ')
    np.savetxt(f, [lambda_target], delimiter=' ', newline=' ')
    np.savetxt(f, [mu_target], delimiter=' ')

    print('nb_loading_cases, loadings', file=f)
    np.savetxt(f, [len(DelFs)], delimiter=' ', newline=' ')
    for i in range(len(DelFs)):
        np.savetxt(f, DelFs[i].flatten(), delimiter=' ', newline=' ')
    print('', file=f)

    print('MPI_size', file=f)
    print(MPI.COMM_WORLD.size, file=f)

### ----- Save initial phase distribution ----- ###
# Cell construction
cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient,
              weights=weights, fft=fft, communicator=MPI.COMM_WORLD)

# Initial phase distribution
phase_ini = np.random.random_sample([cell.nb_quad_pts, *cell.nb_domain_grid_pts])
phase_ini = phase_ini[cell.fft_engine.subdomain_slices]

# Saving
for i in range(cell.nb_quad_pts):
    file_name = folder + f'phase_initial_quad_pt_{i}.npy'
    save_npy(file_name, phase_ini[i], tuple(cell.subdomain_locations),
             tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
phase_ini = phase_ini.flatten(order='F')
phase = phase_ini.copy()

# Copy this file into the folder
helper = 'cp examples/example_with_parallel_l_bfgs.py ' + folder
os.system(helper)

################################################################################
### ----------------------------- Optimization ----------------------------- ###
################################################################################
### ----- Initialisation ----- ###
# Material
phase = phase.reshape([cell.nb_quad_pts, -1], order='F')
mat2 = µ.material.MaterialElasticLocalLame_2d.make(cell, "material")
lame1 = phase ** order * (lambda_1 - lambda_0) + lambda_0
lame2 = phase ** order * (mu_1 - mu_0) + mu_0
for pixel_id, pixel in cell.pixels.enumerate():
    mat2.add_pixel_lame(pixel_id, lame1[:, pixel_id], lame2[:, pixel_id])
cell.initialise()

# Save evolution of aim function
name = folder + 'evolution.txt'
with open(name, 'w') as f:
    title = 'aim_function  norm_sensitivity  average_stresses'
    print(title, file=f)


# Passing function arguments in scipy conform way
opt_args = (cell, mat2, lambda_1, mu_1, lambda_0, mu_0, order,
            DelFs, krylov_solver_args, solver_args, aim_args,
            True, folder)

### ----- Optimization ----- ###
# Possible values for the phase are between 0 and 1
bounds = opt.Bounds(0, 1)

t = time.time()
# With scipy L-BFGS-B optimizer
opt_result = opt.minimize(wrapper, phase.flatten(order='F'), args=opt_args,
                          method='L-BFGS-B', jac=True, bounds=bounds,
                          options={'maxcor': maxcor, 'ftol': ftol, 'gtol': gtol})

################################################################################
### ----------------------------- Save results ----------------------------- ###
################################################################################
### ----- Save metadata of convergence ----- ###
file_name = folder + 'metadata_convergence.txt'
with open(file_name, 'a') as f:
    print('opt_success, opt_time (min), opt_nb_of_iteration', file=f)
    np.savetxt(f, [opt_result.success, (time.time() - t) / 60, opt_result.nit],
               delimiter=' ', newline=' ')
    if not opt_result.success:
        print(file=f)
        print(opt_result.message, file=f)

### ----- Save final phase distribution ----- ###
# Final phase
phase_final = opt_result.x.reshape([cell.nb_quad_pts,
                                    *cell.nb_subdomain_grid_pts], order='F')
file_name = folder + 'phase_last_quad_pt_0.npy'
save_npy(file_name, phase_final [0], tuple(cell.subdomain_locations),
         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)
file_name = folder + 'phase_last_quad_pt_1.npy'
save_npy(file_name, phase_final [1], tuple(cell.subdomain_locations),
         tuple(cell.nb_domain_grid_pts), MPI.COMM_WORLD)

t = time.time() - t
print()
if opt_result.success:
    print('Optimization successful')
else:
    print('Optimization failed')
print('Time (min) =', t/60)
