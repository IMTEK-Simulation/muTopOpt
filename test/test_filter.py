"""
Tests the filter functions
"""
import numpy as np

from muTopOpt.Filter import map_to_unit_range
from muTopOpt.Filter import map_to_unit_range_derivative

def test_map_to_unit_range_derivative(plot=False):
    """ Check the implementation of the derivative of the filter function.
    """
    # Set up
    nb_grid_pts = [5, 4]
    arr = 2 * np.random.random(nb_grid_pts) - 0.5
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]

    # Analytical derivative
    func = map_to_unit_range(arr)
    deriv = map_to_unit_range_derivative(arr)

    # Finite difference derivative
    diff_list = []
    for delta in delta_list:
        deriv_fin_diff = np.empty(deriv.shape)
        for i in range(nb_grid_pts[0]):
            for j in range(nb_grid_pts[1]):
                arr[i, j] += delta
                func_plus = map_to_unit_range(arr)
                deriv_fin_diff[i, j] = (func_plus[i, j] - func[i, j]) / delta
                arr[i, j] -= delta

        diff = np.linalg.norm(deriv_fin_diff - deriv)
        diff_list.append(diff)

    # Fit to linear function
    a = diff_list[0] / delta_list[0]

    # Plot (optional)
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Abs error of derivative of filter-function')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='x', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, a * delta_list, '--', marker='o', label='Fit (lin)')
        ax.legend()
        plt.show()

    assert abs(a * delta_list[1] - diff_list[1]) <= 1e-6
