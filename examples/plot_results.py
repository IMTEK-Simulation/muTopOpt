"""
@file   plot_results.py

@author Indre Jödicke <indre.joedicke@imtek.uni-freiburg.de>

@date   19 Sep 2024

@brief  Postprocessing of optimization results
"""
import sys
import os
#import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['font.cursive'] = ['Arial']
mpl.rcParams['font.size'] = '10'
mpl.rcParams['legend.fontsize'] = '10'
mpl.rcParams['xtick.labelsize'] = '9'
mpl.rcParams['ytick.labelsize'] = '9'
mpl.rcParams['svg.fonttype'] = 'none'

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ
import muFFT

from NuMPI.Optimization import LBFGS
from NuMPI import MPI
from NuMPI.IO import save_npy
from NuMPI.IO import load_npy
from NuMPI.Tools import Reduction

from muTopOpt.Controller import wrapper
from muTopOpt.MaterialDensity import node_to_quad_pt_2_quad_pts_sequential

def node_index(i, j, nb_nodes):
    """
    Turn node coordinates (i, j) into their global node index.

    Parameters
    ----------
    i : int
        x-coordinate (integer) of the node
    j : int
        y-coordinate (integer) of the node
    nb_nodes : tuple of ints
        Number of nodes in the Cartesian directions

    Returns
    -------
    g : int
        Global node index
    """
    Nx, Ny = nb_nodes
    return i + Nx*j

def make_grid(nb_nodes):
    """
    Make an array that contains all elements of the grid. The
    elements are described by the global node indices of
    their corners. The order of the corners is in order of
    the local node index.

    They are sorted in geometric positive order and the first
    is the node with the right angle corner at the bottom
    left. Elements within the same box are consecutive.

    This is the first element per box:

        2
        | \
        |  \
    dy  |   \
        |    \
        0 --- 1

          dx

    This is the second element per box:

           dx
         1 ---0
          \   |
           \  |  dy
            \ |
             \|
              2

    Parameters
    ----------
    nb_nodes : tuple of ints
        Number of nodes in the Cartesian directios

    Returns
    -------
    triangles_el : numpy.ndarray
        Array containing the global node indices of the
        element corners. The first index (suffix _e)
        identifies the element number and the second index
        (suffix _l) the local node index of that element.
    """
    Nx, Ny = nb_nodes
    # These are the node position on a subsection of the grid
    # that excludes the rightmost and topmost nodes. The
    # suffix _G indicates this subgrid.
    y_G, x_G = np.mgrid[:Ny-1, :Nx-1]
    x_G.shape = (-1,)
    y_G.shape = (-1,)

    # List of triangles
    lower_triangles = np.vstack(
        (node_index(x_G, y_G, nb_nodes),
         node_index(x_G+1, y_G, nb_nodes),
         node_index(x_G, y_G+1, nb_nodes)))
    upper_triangles = np.vstack(
        (node_index(x_G+1, y_G+1, nb_nodes),
         node_index(x_G, y_G+1, nb_nodes),
         node_index(x_G+1, y_G, nb_nodes)))
    # Suffix _e indicates global element index
    return np.vstack(
        (lower_triangles, upper_triangles)).T.reshape(-1, 3)

def main(folder):
    """ Function for postprocessing the results of an optimization.
    Parameters
    ----------
    folder: string
            folder in which the results are saved
    """
    ### ----- Read metadata ----- ###
    dim = 2

    # Metadata simulation
    file_name = folder + '/metadata.txt'
    metadata = np.loadtxt(file_name, skiprows=1, max_rows=1)
    nb_grid_pts = metadata[0:2].astype(int)
    Lx, Ly = metadata[2:4]
    eta = metadata[4]
    weight_phase_field = metadata[5]
    lambda_1 = metadata[6]
    mu_1 = metadata[7]
    lambda_0 = metadata[8]
    mu_0 = metadata[9]
    newton_tolerance = metadata[10]
    cg_tolerance = metadata[11]
    equil_tolerance = metadata[12]
    maxiter =int( metadata[13])
    gtol = metadata[14]
    ftol = metadata[15]
    maxcor = metadata[16]
    lambda_target = metadata[17]
    mu_target = metadata[18]

    metadata = np.loadtxt(file_name, skiprows=3, max_rows=1)
    nb_loading = int(metadata[0])
    DelFs = metadata[1:].reshape([nb_loading, 2, 2])

    metadata = np.loadtxt(file_name, skiprows=5, max_rows=1)
    mpi_size = int(metadata)

    # Metadata convergence
    name = folder + 'metadata_convergence.txt'
    if os.path.isfile(name):
        finished = True
        metadata = np.loadtxt(name, skiprows=1, max_rows=1)
        success = bool(metadata[0])
        opt_time = metadata[1]
        nb_iter = int(metadata[2])
    else:
        finished = False
        success = False
        opt_time = 0
        nb_iter = 0

    ### ----- Plot evolution details ----- ###
    name = folder + 'evolution.txt'
    data = np.loadtxt(name, skiprows=1)
    aim = data[:, 0]
    norm_S = data[:, 1]
    average_stresses = data[:, 2:].reshape([-1, nb_loading, dim, dim])

    # Plot evolution of aim_function
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Aim function')
    ax.plot(aim)
    name = folder + '/evolution_of_aim_function.png'
    fig.savefig(name, bbox_inches='tight')

    # Plot evolution of norm_of_sensitivity
    fig, ax = plt.subplots()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Norm of sensitivity')
    ax.plot(norm_S)
    name = folder + '/evolution_of_norm_sensitivity.png'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Print some details of the final result ----- ###
    print('Results:')
    # General success
    if success:
        print('  The optimization finished successfully.')
    elif finished:
        print('  The optimization finished, but did not converge.')
    else:
        print('  The optimization did not finish.')

    # Final value of aim function + average stress
    print(f'  Final aim function = {aim[-1]}')
    for i in range(nb_loading):
        target_stress = 2 * mu_target * DelFs[i]
        trace = DelFs[i][0, 0] + DelFs[i][1, 1]
        target_stress[0, 0] += lambda_target * trace
        target_stress[1, 1] += lambda_target * trace

        print(f'  {i+1}. target stress:', target_stress.flatten())
        print(f'  {i+1}. average stress:', average_stresses[-1, i].flatten())

    # Homogenized material parameters and error to target
    if nb_loading == 1:
        DelF = DelFs[0]
        trace = DelF[0, 0] + DelF[1, 1]
        av_stress = average_stresses[-1, i]
        if (DelF[0, 0] == 0) and (DelF[1, 1] != 0):
            calculated = True
            lambda_hom = av_stress[0, 0] / DelF[1, 1]
            mu_hom = 0.5 * (av_stress[1, 1] / DelF[1, 1] - lambda_hom)
            stress_hom = 2 * mu_hom * DelF + lambda_hom * trace
        elif (DelF[0, 0] != 0) and (DelF[1, 1] != 0):
            if DelF[0, 0] == DelF[1, 1]:
                calculated = False
            else:
                mu_hom = 0.5 * (av_stress[1, 1] - av_stress[0, 0])
                mu_hom = mu_hom / (DelF[1, 1] - DelF[0, 0])
                lambda_hom = av_stress[0, 0] - 2 * mu_hom * DelF[0, 0]
                lambda_hom = lambda_hom / trace
                stress_hom = 2 * mu_hom * DelF + lambda_hom * trace
        else:
            calculated = False

        if calculated:
            if lambda_target == 0:
                print(f'  First lame constant: target={lambda_target}',
                      f'hom={lambda_hom:.3g}')
            else:
                err = (lambda_hom - lambda_target) / lambda_target * 100
                print(f'  First lame constant: target={lambda_target}',
                      f'hom={lambda_hom:.3g}    error={err:.3g}%')
            err = (mu_hom - mu_target) / mu_target * 100
            print(f'  Second lame constant: target={mu_target}',
                  f'hom={mu_hom:.3g}    error={err:.3g}%')
            err = np.linalg.norm(stress_hom - av_stress)
            err = err / np.linalg.norm(av_stress) * 100
            print(f'  Difference between average stress and isotropic stress',
                  f'calculated with hom material param = {err:.3g} %')
        else:
            message = 'Attention: The homogenized material parameters '
            message += 'could not be calculated. The strain is '
            message += f'[[{DelF[0, 0]}, {DelF[0, 1]}], [{DelF[1, 0]}, {DelF[1, 1]}]]'
            print(message)

    ### ----- Preparations for plotting phases ----- ###
    # How many cells will be plotted
    nb_cells = [5, 3]

    # Should the pictures be rasterized? -> Much faster if rasterized
    rasterized = True

    # Helper definitions for plotting
    hx = Lx / nb_grid_pts[0]
    hy = Ly / nb_grid_pts[1]

    # Coordinates of pixels
    x = np.linspace(0, nb_cells[0] * Lx, nb_cells[0] * nb_grid_pts[0] + 1,
                    endpoint=True)
    y = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] * nb_grid_pts[1] + 1,
                    endpoint=True)

    # Coordinates of unit cell
    x_cells = np.linspace(0, nb_cells[0] * Lx, nb_cells[0] + 1, endpoint=True)
    y_cells = np.linspace(0, nb_cells[1] * Ly, nb_cells[1] + 1, endpoint=True)

    # Triangulation
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    triangles = make_grid([nb_cells[0] * nb_grid_pts[0] + 1,
                           nb_cells[1] * nb_grid_pts[1] + 1])
    triangulation = mpl.tri.Triangulation(X, Y, triangles)

    ### ----- Plot initial phase ----- ###
    print('Plotting initial phase.')
    # Read data
    #file_name = folder + '/phase_initial_quad_pt_0.npy'
    #phase_ini_0 = np.load(file_name)
    #file_name = folder + '/phase_initial_quad_pt_1.npy'
    #phase_ini_1 = np.load(file_name)
    #phase_ini = np.stack((phase_ini_0, phase_ini_1), axis=0)
    file_name = folder + '/phase_initial.npy'
    phase_ini = np.load(file_name)
    phase_ini = node_to_quad_pt_2_quad_pts_sequential(phase_ini)

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

    # Plot phase distribution
    phase_plot = phase_ini.copy()
    for i in range(1, nb_cells[0]):
        phase_plot = np.concatenate((phase_plot, phase_ini), axis=1)
    helper = phase_plot
    for i in range(1, nb_cells[1]):
        phase_plot = np.concatenate((phase_plot, helper), axis=2)
    phase_plot = phase_plot.flatten(order='F')
    tpc = ax.tripcolor(triangulation, phase_plot, cmap='jet',
                       vmin=0, vmax=1, rasterized=rasterized)

    # Plot unit cells outline
    phase_plot = np.zeros((nb_cells[1], nb_cells[0]))
    mask = np.ma.masked_array(phase_plot, mask=True)
    ax.pcolormesh(x_cells, y_cells, mask, alpha=1, edgecolor='black',
                  rasterized=rasterized)

    # Colorbar
    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)

    cbar = fig.colorbar(tpc, cax=ax_cb)
    cbar.ax.set_ylabel(r'Phase $\rho$', rotation=90, labelpad=10)

    # Save figure
    name = folder + '/phase_initial.pdf'
    fig.savefig(name, bbox_inches='tight')

    ### ----- Plot last phase ----- ###
    print('Plotting last phase.')
    # Read data
    file_name = folder + '/phase_last.npy'
    phase_last = np.load(file_name)
    phase_last = node_to_quad_pt_2_quad_pts_sequential(phase_last)

    # Prepare figure
    fig = plt.figure()
    if success:
        fig.suptitle('Optimized material density')
    else:
        fig.suptitle('Material density of last optimization step')
    gs = fig.add_gridspec(nrows=1, ncols=1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')
    ax.set_xlabel('Position x')
    ax.set_ylabel('Position y')
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    # Plot phase distribution
    phase_plot = phase_last.copy()
    for i in range(1, nb_cells[0]):
        phase_plot = np.concatenate((phase_plot, phase_last), axis=1)
    helper = phase_plot
    for i in range(1, nb_cells[1]):
        phase_plot = np.concatenate((phase_plot, helper), axis=2)
    phase_plot = phase_plot.flatten(order='F')
    tpc = ax.tripcolor(triangulation, phase_plot, cmap='jet',
                       vmin=0, vmax=1, rasterized=rasterized)

    # Plot unit cells outline
    phase_plot = np.zeros((nb_cells[1], nb_cells[0]))
    mask = np.ma.masked_array(phase_plot, mask=True)
    ax.pcolormesh(x_cells, y_cells, mask, alpha=1, edgecolor='black',
                  rasterized=rasterized)

    # Colorbar
    divider = make_axes_locatable(ax)

    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)

    cbar = fig.colorbar(tpc, cax=ax_cb)
    cbar.ax.set_ylabel(r'Material density $\rho$', rotation=90, labelpad=10)

    # Save figure
    if success:
        name = folder + '/phase_optimized.pdf'
    else:
        name = folder + '/phase_last.pdf'
    fig.savefig(name, bbox_inches='tight')




if __name__ == "__main__":
    folder = sys.argv[1]
    main(folder)
