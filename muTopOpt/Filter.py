"""
This file contains filter functions
"""
import numpy as np

def map_to_unit_range(arr):
    """ Map an array to the intervall [0, 1].
    Parameter
    ---------
    arr: np.ndarray of floats
         Values to map

    Retunrs
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

    Retunrs
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
