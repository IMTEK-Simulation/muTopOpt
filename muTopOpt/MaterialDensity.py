"""
This file contains the functions to calculate the material
density from the phase field.
"""
import sys
import os
import numpy as np

# Default path of the library
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmufft/python"))
sys.path.insert(0, os.path.join(os.getcwd(), "../muspectre/builddir/language_bindings/libmugrid/python"))
import muSpectre as µ

def map_to_unit_range(arr):
    """ Map an array to the intervall [0, 1].
    Parameter
    ---------
    arr: np.ndarray of floats
         Values to map

    Returns
    -------
    mapped: np.ndarray(arr.shape) of floats
            Mapped values
    """
    threshold_down = arr < 0
    threshold_up = arr > 1
    mapped = 1 - (8 + 9 * np.cos(np.pi * arr) -\
                          np.cos(3 * np.pi * arr)) / 16
    mapped[threshold_down] = 0
    mapped[threshold_up] = 1
    return mapped


def map_to_unit_range_derivative(arr):
    """ Derivative of map_to_unit_range with respect to arr.
    Parameter
    ---------
    arr: np.ndarray of floats
         Values to map

    Returns
    -------
    derivative: np.ndarray(arr.shape) of floats
                Derivatives with respect to arr
    """
    threshold_down = arr < 0
    threshold_up = arr > 1
    derivative = np.sin(3 * np.pi * arr) - 3 * np.sin(np.pi * arr)
    derivative = - 3 * np.pi / 16 * derivative
    derivative[threshold_down] = 0
    derivative[threshold_up] = 0
    return derivative

def node_to_quad_pt_2_quad_pts_sequential(phase):
    """ Interpolate a field defined at the nodes to the
        quadrature points for a discretization of linear
        rectangular triangles.
        This implementation works only sequentially.
    Parameter
    ---------
    phase: np.ndarray (nb_grid_pts) of floats
           Values of the field to interpolate

    Returns
    -------
    material: np.ndarray([2, nb_grid_pts) of floats
              Interpolated field at the 2 quadrature points
    """
    if len(phase.shape) != 2:
        message = f'The phase field has the shape {phase.shape} '
        message += f'but it must be of the shape nb_grid_pts.'
        raise ValueError(message)
    material0 = np.roll(phase, -1, axis=0) + np.roll(phase, -1, axis=1)
    material0 = (material0 + phase) / 3
    material1 = np.roll(phase, -1, axis=0) + np.roll(phase, -1, axis=1)
    material1 = (material1 + np.roll(phase, [-1, -1], axis=(0, 1))) / 3

    return np.stack((material0, material1), axis=0)

def node_to_quad_pt_2_quad_pts(phase, cell):
    """ Interpolate a field defined at the nodes to the
        quadrature points for a discretization of linear
        rectangular triangles.
    Parameter
    ---------
    phase: np.ndarray (nb_grid_pts) of floats
           Values of the field to interpolate

    Returns
    -------
    material: np.ndarray([2, nb_grid_pts) of floats
              Interpolated field at the 2 quadrature points
    """
    if len(phase.shape) != 2:
        message = f'The phase field has the shape {phase.shape} '
        message += f'but it must be of the shape nb_grid_pts.'
        raise ValueError(message)
    fft = cell.fft_engine
    q = fft.fftfreq
    fourier_phase = fft.fetch_or_register_fourier_space_field(
        "fft_workspace", 1)
    fft.fft(phase, fourier_phase)

    # Lower left triangle
    #d = µ.DiscreteDerivative([0, 0], [[[1], [1]], [[1], [0]]])
    d = µ.DiscreteDerivative([0, 0], [[1, 0], [1, 1]])
    material0 = np.zeros_like(phase)
    d = d.fourier(q)
    fft.ifft(d * fourier_phase, material0)
    material0 *= fft.normalisation

    #material0 = np.roll(phase, -1, axis=0) + np.roll(phase, -1, axis=1)
    #material0 = (material0 + phase) / 3
    # Upper right triangle
    material1 = np.roll(phase, -1, axis=0) + np.roll(phase, -1, axis=1)
    material1 = (material1 + np.roll(phase, [-1, -1], axis=(0, 1))) / 3

    return np.stack((material0, material1), axis=0)

def df_dphase_2_quad_pts_derivative_sequential(df_dmat):
    """ Derivative of a function with respect to the phase
        defined at the nodes. The interpolation from nodes to quadrature
        points is done with linear finite elements using
        rectangular triangles.
        This implementation works only sequentially.
    Parameter
    ---------
    phase: np.ndarray (nb_grid_pts) of floats
           Values of the phase-field at the nodes
    df_dmat: np.ndarray([-1, nb_quad_pts, *nb_grid_pts]) of floats
             Derivative of the function with respect to the
             material density defined at the quadrature points.

    Returns
    -------
    df_dphase: np.ndarray([-1, nb_grid_pts]) of floats
               Derivative of the function with respect to the phase.
    """
    if (len(df_dmat.shape) != 4):
        message = f'df_dmat has the shape {df_dmat.shape} '
        message += f'but it must be of the shape of [-1, nb_quad_pts, *nb_grid_pts].'
        raise ValueError(message)
    if (df_dmat.shape[1] != 2):
        message = f'The number of quadrature is {df_dmat.shape[1]}, but '
        message += 'it must be 2.'
        raise ValueError(message)
    df_dphase = np.empty([df_dmat.shape[0], *df_dmat.shape[2:]])
    for i in range(df_dmat.shape[0]):
        helper = np.roll(df_dmat[i, 1, :, :], [1, 1], axis=(0, 1))
        helper += np.roll(df_dmat[i, 0, :, :], 1, axis=1)
        helper += np.roll(df_dmat[i, 1, :, :], 1, axis=1)
        helper += np.roll(df_dmat[i, 0, :, :], 1, axis=0)
        helper += np.roll(df_dmat[i, 1, :, :], 1, axis=0)
        helper += df_dmat[i, 0, :, :]
        df_dphase[i] = helper / 3

    return df_dphase
