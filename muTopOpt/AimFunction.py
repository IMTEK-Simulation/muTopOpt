"""
This file contains the complete aim function and its partial
derivatives for a topology optimization with a target stress
and a phase field regularization.
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

import muTopOpt.StressTarget as st
import muTopOpt.PhaseField as pf
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative_sequential
from muTopOpt.MaterialDensity import df_dphase_2_quad_pts_derivative

def aim_function_sequential(cell, phase, strains, stresses, target_stresses, eta,
                 weight_phase_field):
    aim = st.square_error_target_stresses_sequential(cell, strains,
                                                     stresses,
                                                     target_stresses)
    if cell.nb_quad_pts == 2:
        aim += weight_phase_field * pf.phase_field_energy_sequential(phase, cell, eta)
    else:
        aim += weight_phase_field * pf.phase_field_rectangular_grid(phase, eta, cell)
    return aim

def aim_function(cell, phase, strains, stresses, target_stresses, eta,
                 weight_phase_field):
    aim = st.square_error_target_stresses(cell, strains,
                                          stresses,
                                          target_stresses)
    if cell.nb_quad_pts == 2:
        aim += weight_phase_field * pf.phase_field_energy(phase, cell, eta)
    else:
        aim += weight_phase_field * pf.phase_field_rectangular_grid(phase, eta, cell)
    return aim

def aim_function_deriv_strain_sequential(cell, strains, stresses, target_stresses,
                              eta, weight_phase_field):
    aim_deriv_strain =\
        st.square_error_target_stresses_deriv_strains_sequential(cell, strains,
                                                                 stresses,
                                                                 target_stresses)
    return aim_deriv_strain

def aim_function_deriv_strain(cell, strains, stresses, target_stresses,
                              eta, weight_phase_field):
    aim_deriv_strain =\
        st.square_error_target_stresses_deriv_strains(cell, strains, stresses,
                                                      target_stresses)
    return aim_deriv_strain

def aim_function_deriv_phase_sequential(cell, phase, strains, stresses,
                             dstress_dmat_list, target_stresses,
                             eta, weight_phase_field):

    if cell.nb_quad_pts == 2:
        helper = st.square_error_target_stresses_deriv_phase(cell, stresses,
                                                             target_stresses,
                                                             dstress_dmat_list)
        helper = helper.reshape([-1, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts], order='F')
        aim_deriv_phase = df_dphase_2_quad_pts_derivative_sequential(helper)
        aim_deriv_phase = aim_deriv_phase.reshape(cell.nb_subdomain_grid_pts)
        aim_deriv_phase +=\
            weight_phase_field * pf.phase_field_energy_deriv_sequential(phase, cell, eta)
    else:
        aim_deriv_phase = st.square_error_target_stresses_deriv_phase(cell, stresses,
                                                                      target_stresses,
                                                                      dstress_dmat_list)
        aim_deriv_phase +=\
            weight_phase_field * pf.phase_field_rectangular_grid_deriv_phase(phase, eta, cell)

    return aim_deriv_phase

def aim_function_deriv_phase(cell, phase, strains, stresses,
                             dstress_dmat_list, target_stresses,
                             eta, weight_phase_field):

    if cell.nb_quad_pts == 2:
        helper = st.square_error_target_stresses_deriv_phase(cell, stresses,
                                                             target_stresses,
                                                             dstress_dmat_list)
        helper = helper.reshape([-1, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts], order='F')
        aim_deriv_phase = df_dphase_2_quad_pts_derivative(helper, cell)
        aim_deriv_phase = aim_deriv_phase.reshape(cell.nb_subdomain_grid_pts)
        aim_deriv_phase +=\
            weight_phase_field * pf.phase_field_energy_deriv(phase, cell, eta)
    else:
        aim_deriv_phase = st.square_error_target_stresses_deriv_phase(cell, stresses,
                                                                      target_stresses,
                                                                      dstress_dmat_list)
        aim_deriv_phase +=\
            weight_phase_field * pf.phase_field_rectangular_grid_deriv_phase(phase, eta, cell)

    return aim_deriv_phase
