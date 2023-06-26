"""
@file   plot_results.py

@author Indre Jödicke <indre.joedicke@imtek.uni-freiburg.de>

@date   26 Jun 2023

@brief  Plot results of an optimization
"""
#import sys
import os
#sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/python"))
#sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmufft/python"))
#sys.path.insert(0, os.path.join(os.getcwd(), "./muspectre/build/language_bindings/libmugrid/python"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from time import time

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '10'
mpl.rcParams['legend.fontsize'] = '10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'

import muSpectre as µ
import muFFT

from muTopOpt.Filter import map_to_unit_range
from muTopOpt.StressTarget import square_error_target_stresses
from muTopOpt.PhaseField import phase_field_rectangular_grid
from muTopOpt.PhaseField import gradient_rectangular_grid
from muTopOpt.PhaseField import double_well_potential

### ----- Define the hexagonal grid ----- ###
def main(folder, three_loadings=False, equilat_triangle_grid=False):
    """ Function for postprocessing the results of an optimization.
    Parameters
    ----------
    folder: string
            folder in which the results are saved
    three_loadings: boolean
                    If True three load cases have been considered.
                    If False (default) one load case has been considered.
    equilat_triangle_grid: boolean
                           If True equilateral triangles are used as discretization.
                           If False rectangular triangles are used as discretization.
                           Default is False.
    """
    if equilat_triangle_grid:
        print('NOTE: Discretization with equilateral triangles.')
    if three_loadings:
        print('NOTE: Three loadings')
    else:
        print('NOTE: One loading')
    print()

    ### ----- Plot evolution of aim function ----- ###
    aim_function = None
    print('Plotting evolution of aim function.')
    # Read data
    file_name = folder + '/evolution_of_aim_function.txt'
    aim_function = np.loadtxt(file_name)

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Aim function')
    ax.plot(aim_function)
    name = folder + '/evolution_of_aim_function.png'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Plot details of aim function ----- ###
    file_name = folder + '/evolution_of_aim_function_details.txt'
    data = np.loadtxt(file_name, skiprows=1)
    sq_err = data[:, 1]
    ph_field = data[:, 2]
    norm_S = data[:, 3]

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Function')
    ax.plot(sq_err, label='error_target_stress')
    ax.plot(ph_field, label='weighted phase_field')
    ax.legend()
    name = folder + '/evolution_of_aim_function_details.png'
    fig.savefig(name, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Norm of sensitivity')
    ax.plot(norm_S)
    name = folder + '/evolution_of_norm_sensitivity.png'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Read metadata ----- ###
    # Did the optimization finish?
    file_name = folder + '/phase_final.npy'
    if os.path.isfile(file_name):
        finished = True
        phase = np.load(file_name)
    else:
        finished = False
        file_name = folder + '/phase_last_step.npy'
        phase = np.load(file_name)

    # Metadata simulation
    file_name = folder + '/metadata.txt'
    metadata = np.loadtxt(file_name, skiprows=1, max_rows=1)
    nb_grid_pts = metadata[0:2].astype(int)
    Lx, Ly = metadata[2:4]
    weight = metadata[4]
    eta = metadata[5]
    Young1 = metadata[6]
    Poisson1 = metadata[7]
    Young2 = metadata[8]
    Poisson2 = metadata[9]
    newton_tolerance = metadata[10]
    cg_tolerance = metadata[11]
    equil_tolerance = metadata[12]
    maxiter =int( metadata[13])
    loading = metadata[17]
    target_shear_modulus = metadata[18]
    target_lame2 = metadata[19]
    nb_strain_steps = int(metadata[20])
    metadata = np.loadtxt(file_name, skiprows=3, max_rows=1)
    mpi_size = int(metadata)

    # Metadata convergence of optimizer
    if finished:
        file_name = folder + '/metadata_convergence.txt'
        if os.path.isfile(file_name):
            metadata = np.loadtxt(file_name, skiprows=1, max_rows=1)
        else:
            exit(file_name + ' is no file.')
        success = bool(metadata[0])
        opt_time = metadata[1]
        nb_iter = int(metadata[2])

    ### ----- Preparations for plotting phase ----- ###
    # How many cells will be plotted
    if equilat_triangle_grid:
        nb_cells = [3, 3]
    else:
        nb_cells = [5, 3]

    # Should the pictures be rasterized? -> Much faster if rasterized
    rasterized = True

    # Helper definitions for plotting
    hx = Lx / nb_grid_pts[0]
    hy = Ly / nb_grid_pts[1]

    if equilat_triangle_grid:
        delta_x = 0.5 * hx * nb_grid_pts[1]
        xmax = nb_cells[0] * Lx + nb_cells[1] * delta_x
        # plot additional cells to fill up a rectangle
        nb_add_cells = int( - ((- delta_x * nb_cells[1]) // Lx))
        nb_cells[0] = nb_cells[0] + 2 * nb_add_cells
        # Coordinates of pixels
        x = np.linspace(-nb_add_cells * Lx, (nb_cells[0] - nb_add_cells) * Lx,
                        nb_cells[0] * nb_grid_pts[0] + 1, endpoint=True)
        y = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] * nb_grid_pts[1] + 1,
                        endpoint=True)
        x, y = np.meshgrid(x, y)
        x = x + 0.5 * hx / hy * y
        # Coordinates of unit cell
        x_cells = np.linspace(-nb_add_cells * Lx, (nb_cells[0] - nb_add_cells) * Lx,
                              nb_cells[0] + 1, endpoint=True)
        y_cells = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] + 1, endpoint=True)
        x_cells, y_cells = np.meshgrid(x_cells, y_cells)
        x_cells = x_cells + delta_x / Ly * y_cells
    else:
        # Coordinates of pixels
        x = np.linspace(0, nb_cells[0] * Lx, nb_cells[0] * nb_grid_pts[0] + 1,
                        endpoint=True)
        y = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] * nb_grid_pts[1] + 1,
                        endpoint=True)
        # Coordinates of unit cell
        x_cells = np.linspace(0, nb_cells[0] * Lx, nb_cells[0] + 1, endpoint=True)
        y_cells = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] + 1, endpoint=True)

    ### ----- Plot initial phase ----- ###
    print('Plotting initial phase.')
    # Read data
    file_name = folder + '/phase_ini.npy'
    phase_ini = np.load(file_name)

    # Prepare figure
    fig = plt.figure()
    fig.suptitle('Initial material density')
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Position y')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if equilat_triangle_grid:
        ax.set_xlim(0, xmax)

    # Plot phase distribution
    phase_plot = phase_ini
    for i in range(1, nb_cells[0]):
        phase_plot = np.concatenate((phase_plot, phase_ini), axis=0)
    helper = phase_plot
    for i in range(1, nb_cells[1]):
        phase_plot = np.concatenate((phase_plot, helper), axis=1)
    phase_plot = phase_plot.transpose()
    phase_plot = map_to_unit_range(phase_plot)
    tpc = ax.pcolormesh(x, y, phase_plot, cmap='jet', vmin=0, vmax=1, rasterized=rasterized)

    # Plot unit cells outline
    phase_plot = np.zeros((nb_cells[1], nb_cells[0]))
    mask = np.ma.masked_array(phase_plot, mask=True)
    ax.pcolormesh(x_cells, y_cells, mask, alpha=1, edgecolor='black', rasterized=rasterized)

    # Colorbar
    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)

    cbar = fig.colorbar(tpc, cax=ax_cb)
    cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

    # Save figure
    name = folder + '/phase_initial.pdf'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Plot final phase ----- ###
    print('Plotting final phase.')
    # Prepare figure
    fig = plt.figure(dpi=700)
    if finished:
        if success:
            fig.suptitle('Optimized filtered phase')
        else:
            fig.suptitle('Optimized filtered phase (opt not successful)')
    else:
        fig.suptitle('Filtered phase of last opt step')
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Position y')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    if equilat_triangle_grid:
        ax.set_xlim(0, xmax)

    # Plot phase distribution
    phase_plot = phase
    for i in range(1, nb_cells[0]):
        phase_plot = np.concatenate((phase_plot, phase), axis=0)
    helper = phase_plot
    for i in range(1, nb_cells[1]):
        phase_plot = np.concatenate((phase_plot, helper), axis=1)
    phase_plot = phase_plot.transpose()
    phase_plot = map_to_unit_range(phase_plot)
    tpc = ax.pcolormesh(x, y, phase_plot, cmap='jet', vmin=0, vmax=1, rasterized=rasterized)

    # Plot unit cells outline
    phase_plot = np.zeros((nb_cells[1], nb_cells[0]))
    mask = np.ma.masked_array(phase_plot, mask=True)
    ax.pcolormesh(x_cells, y_cells, mask, alpha=1, edgecolor='black', rasterized=rasterized)

    # Colorbar
    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)

    cbar = fig.colorbar(tpc, cax=ax_cb)
    cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

    # Save figure
    name = folder + '/phase_last_step.pdf'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Print metadata ----- ###
    print('--- Metadata problem ---')
    if three_loadings:
        if equilat_triangle_grid:
            print(f'Optimize for isotrop material (equilat_triangles).')
        else:
            print(f'Optimize for isotrop material (rectangular_triangles).')
    else:
        print(f'Optimize for negative Poissons ratio.')
    print(f'nb_grid_pts = {nb_grid_pts[0]} x {nb_grid_pts[1]}')
    print(f'weight = {weight}')
    print(f'eta = {eta}')
    print(f'target_shear_modulus = {target_shear_modulus}')
    print(f'target_lame2 = {target_lame2}')

    # Metadata of restart
    file_name = folder + '/metadata.txt'
    with open(file_name) as f:
            for line in f:
                pass
            last_line = line
            if last_line[0] == 'I':
                print(last_line)

    print()
    print('--- Metadata muSpectre ---')
    print(f'newton_tolerance = {newton_tolerance}')
    print(f'cg_tolerance = {cg_tolerance}')
    print(f'mpi_size = {mpi_size}')

    print()
    print('--- Metadata convergence ---')
    if finished and success:
        print('Optimization successful.')
        print(f'time (min) = {opt_time}')
        print(f'nb_iter = {nb_iter}')
    elif finished:
        print('Optimization did not succeed.')
        file_name = folder + '/metadata_convergence.txt'
        with open(file_name) as f:
            for line in f:
                pass
            last_line = line
        print('Message:', last_line)
        print(f'time (min) = {opt_time}')
        print(f'nb_iter = {nb_iter}')
    else:
        print('Optimization aborted.')
    if aim_function is not None:
        print(f'initial aim function = {aim_function[0]}')
        print(f'last aim function = {aim_function[-1]}')

    ### ----- Calculate interesting data for the last phase ----- ###
    # Additional parameters for muSpectre
    formulation = µ.Formulation.small_strain
    fft = 'mpi'
    lengths = [Lx, Ly]
    dim = len(nb_grid_pts)
    if equilat_triangle_grid:
        d1x = muFFT.DiscreteDerivative([0, 0], [[-1], [1]])
        d1y = muFFT.DiscreteDerivative([0, 0], [[-1/2, 1], [-1/2, 0]])
        d2x = muFFT.DiscreteDerivative([0, 0], [[0, -1], [0, 1]])
        d2y = muFFT.DiscreteDerivative([0, 0], [[0, 1/2], [-1, 1/2]])
        gradient = [d1x, d1y, d2x, d2y]
    else:
        gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                    muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    if three_loadings:
        DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim]), np.zeros([dim, dim])]
        DelFs[0][0, 0] = loading
        DelFs[1][1, 1] = loading
        DelFs[2][0, 1] = loading/2
        DelFs[2][1, 0] = loading/2
    else:
        DelFs = [np.zeros([dim, dim])]
        DelFs[0][1, 1] = loading
    verbose = µ.Verbosity.Silent
    krylov_solver_args = (cg_tolerance, maxiter, verbose)
    solver_args = (newton_tolerance, equil_tolerance, verbose)

    # Calculate aimed for stresses
    Young_aim = target_shear_modulus * (3*target_lame2 + 2*target_shear_modulus)
    Young_aim = Young_aim / (target_lame2 + target_shear_modulus)
    Poisson_aim = 0.5 * target_lame2 / (target_lame2 + target_shear_modulus)
    cell = µ.Cell([1, 1], lengths, formulation, fft=fft)
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    mat.add_pixel(0, Young_aim, Poisson_aim)

    target_stresses = []
    for DelF in DelFs:
        strain = DelF
        strain = strain.reshape([*DelF.shape, 1, 1], order='F')
        target_stress = cell.evaluate_stress(strain)
        target_stress = np.average(target_stress, axis=(2, 3, 4))
        # Nondimensionalization
        target_stress = target_stress / Young2 / loading
        target_stresses.append(target_stress)

    # muSpectre initialization
    phase = phase.flatten(order='F')
    Young = (Young2 - Young1) * map_to_unit_range(phase) + Young1
    Poisson = (Poisson2 - Poisson1) * map_to_unit_range(phase) + Poisson1
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, fft=fft, weights=[1, 1])
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id, pixel in cell.pixels.enumerate():
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # Solve the equilibrium equations
    shape = [dim, dim, cell.nb_quad_pts, *nb_grid_pts]
    krylov_solver = µ.solvers.KrylovSolverCG(cell, *krylov_solver_args)
    strains = []
    stresses = []
    average_stresses = []
    for DelF in DelFs:
        applied_strain = []
        for s in range(1, nb_strain_steps+1):
            applied_strain.append(s / nb_strain_steps * DelF)
        result = µ.solvers.newton_cg(cell, applied_strain, krylov_solver,
                                     *solver_args)
        strain = result[nb_strain_steps-1].grad.reshape(shape, order='F').copy()
        strains.append(strain)
        stress = cell.evaluate_stress(strain).copy()
        stresses.append(stress)
        stress = np.average(stress, axis=(2, 3, 4))
        average_stresses.append(stress)

    # Calculate the aim function
    ph_field_grad = gradient_rectangular_grid(phase, cell)
    ph_field_pot = double_well_potential(phase, cell)
    ph_field = phase_field_rectangular_grid(phase, eta, cell)
    sq_err = square_error_target_stresses(cell, strains, stresses, target_stresses,
                                          loading, Young2)
    aim = sq_err + weight * ph_field

    # Print
    print()
    print('--- Contributions to the aim function (last step) ---')
    print(f'Aim function = {aim}')
    print(f'Square error = {sq_err}')
    print(f'Weighted phase field = {weight * ph_field}')
    print(f'Phase field = {ph_field}')
    print(f'Phase field (weighted gradient) = {eta * ph_field_grad}')
    print(f'Phase field (weighted double-well pot) = {1/eta * ph_field_pot}')
    print(f'Phase field (gradient) = {ph_field_grad}')
    print(f'Phase field (double-well pot) = {ph_field_pot}')

    ### ----- Homogenized material constants ----- ###
    if three_loadings:
        # Calculate homogenized stiffness tensor from stresses
        K_hom = np.empty((dim, dim, dim, dim))
        stress0 = average_stresses[0]
        stress1 = average_stresses[1]
        stress2 = average_stresses[2]
        strain0 = DelFs[0]
        strain1 = DelFs[1]
        strain2 = DelFs[2]
        K_hom[0, 0, 0, 0] = stress0[0, 0] / strain0[0, 0]
        K_hom[1, 1, 0, 0] = stress0[1, 1] / strain0[0, 0]
        K_hom[0, 1, 0, 0] = K_hom[1, 0, 0, 0] = stress0[0, 1] / strain0[0, 0]
        K_hom[0, 0, 1, 1] = stress1[0, 0] / strain1[1, 1]
        K_hom[1, 1, 1, 1] = stress1[1, 1] / strain1[1, 1]
        K_hom[0, 1, 1, 1] = K_hom[1, 0, 1, 1] = stress1[0, 1] / strain1[1, 1]
        K_hom[0, 0, 0, 1] = K_hom[0, 0, 1, 0] = 0.5 * stress2[0, 0] / strain2[0, 1]
        K_hom[1, 1, 0, 1] = K_hom[1, 1, 1, 0] = 0.5 * stress2[1, 1] / strain2[0, 1]
        K_hom[0, 1, 0, 1] = K_hom[0, 1, 1, 0] = 0.5 * stress2[0, 1] / strain2[0, 1]
        K_hom[1, 0, 0, 1] = K_hom[1, 0, 1, 0] = 0.5 * stress2[1, 0] / strain2[0, 1]

        # Material constants
        mu_num = K_hom[1, 0, 1, 0]
        lambda_num = K_hom[0, 0, 1, 1]

        # Target stiffness tensor
        K_target = np.zeros((dim, dim, dim, dim))
        K_target[0, 0, 0, 0] = K_target[1, 1, 1, 1] = 2 * target_shear_modulus + target_lame2
        K_target[0, 1, 1, 0] = target_shear_modulus
        K_target[0, 1, 0, 1] = K_target[1, 0, 0, 1] = K_target[1, 0, 1, 0] = K_target[0, 1, 1, 0]
        K_target[0, 0, 1, 1] = K_target[1, 1, 0, 0] = target_lame2

    else:
        lambda_num = average_stresses[0][0, 0] / DelFs[0][1, 1]
        mu_num = 0.5 * (average_stresses[0][1, 1] / DelFs[0][1, 1] - lambda_num)

    # Printing
    print()
    print('--- Homogenized material constants ---')
    print(f'lambda_target = {target_lame2}     lambda_num = {lambda_num}')
    print(f'mu_target = {target_shear_modulus}     mu_num = {mu_num}')
    if three_loadings:
        print('K_hom =', K_hom.flatten())
        print('K_target =', K_target.flatten())
        print(f'diff K_hom and K_target = {np.linalg.norm(K_hom - K_target)}')

if __name__ == "__main__":
    import sys
    folder = sys.argv[1]
    main(folder, three_loadings=True,
         equilat_triangle_grid=True)
