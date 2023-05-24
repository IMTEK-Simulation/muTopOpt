"""
Tests the derivative of the stress with respect to the material density.
"""
import numpy as np
import muSpectre as µ
import muFFT
#from muTopOpt import sensitivity_analysis as sa
from muSpectre import sensitivity_analysis as sa

def test_dstress_dphase(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_Young = 10.
    delta_Poisson = 0.3

    # Load cases
    #DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    #DelFs[0][0, 0] = 0.01
    #DelFs[1][0, 1] = 0.007 / 2
    #DelFs[1][1, 0] = 0.007 / 2
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Material initialization
    Young = delta_Young * phase
    Poisson = delta_Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    derivs = sa.calculate_dstress_dphase(cell, strains, Young, delta_Young,
                                         Poisson, delta_Poisson)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2)
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            #for i in range(len(phase)):
            for pixel_id in cell.pixel_indices:
                # New material
                helper_cell = µ.Cell(nb_grid_pts, lengths, formulation)
                mat = µ.material.MaterialLinearElastic4_2d.make(helper_cell, "material")
                for iter_pixel, pixel in cell.pixels.enumerate():
                    if pixel_id == iter_pixel:
                        mat.add_pixel(iter_pixel, Young[pixel_id] + delta_Young * delta,
                                      Poisson[pixel_id] + delta_Poisson * delta)
                    else:
                        mat.add_pixel(iter_pixel, Young[pixel_id], Poisson[pixel_id])

                # Derivative
                stress_plus = helper_cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] = (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta

            diff += np.linalg.norm(deriv[:, :, :, pixel_id] - deriv_fin_diff[:, :, :, pixel_id])**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6


def test_dstress_dphase_void(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_Young = 10.
    delta_Poisson = 0.3
    phase[0] = 0

    # Load cases
    #DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    #DelFs[0][0, 0] = 0.01
    #DelFs[1][0, 1] = 0.007 / 2
    #DelFs[1][1, 0] = 0.007 / 2
    DelFs = [np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
        # delta_list = [1e-5]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Material initialization
    Young = delta_Young * phase
    Poisson = delta_Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    derivs = sa.calculate_dstress_dphase(cell, strains, Young, delta_Young,
                                         Poisson, delta_Poisson)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2)
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            #for i in range(len(phase)):
            for pixel_id in cell.pixel_indices:
                # New material
                helper_cell = µ.Cell(nb_grid_pts, lengths, formulation)
                mat = µ.material.MaterialLinearElastic4_2d.make(helper_cell, "material")
                for iter_pixel, pixel in cell.pixels.enumerate():
                    if pixel_id == iter_pixel:
                        mat.add_pixel(iter_pixel, Young[pixel_id] + delta_Young * delta,
                                      Poisson[pixel_id] + delta_Poisson * delta)
                    else:
                        mat.add_pixel(iter_pixel, Young[pixel_id], Poisson[pixel_id])

                # Derivative
                stress_plus = helper_cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] = (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta

            diff += np.linalg.norm(deriv[:, :, :, pixel_id] - deriv_fin_diff[:, :, :, pixel_id])**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6

def test_dstress_dphase_two_strains(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain
    cell = µ.Cell(nb_grid_pts, lengths, formulation)

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_Young = 10.
    delta_Poisson = 0.3
    phase[0] = 0

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
        # delta_list = [1e-5]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    # Material initialization
    Young = delta_Young * phase
    Poisson = delta_Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    derivs = sa.calculate_dstress_dphase(cell, strains, Young, delta_Young,
                                         Poisson, delta_Poisson)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2)
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            #for i in range(len(phase)):
            for pixel_id in cell.pixel_indices:
                # New material
                helper_cell = µ.Cell(nb_grid_pts, lengths, formulation)
                mat = µ.material.MaterialLinearElastic4_2d.make(helper_cell, "material")
                for iter_pixel, pixel in cell.pixels.enumerate():
                    if pixel_id == iter_pixel:
                        mat.add_pixel(iter_pixel, Young[pixel_id] + delta_Young * delta,
                                      Poisson[pixel_id] + delta_Poisson * delta)
                    else:
                        mat.add_pixel(iter_pixel, Young[pixel_id], Poisson[pixel_id])

                # Derivative
                stress_plus = helper_cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] = (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta

            diff += np.linalg.norm(deriv[:, :, :, pixel_id] - deriv_fin_diff[:, :, :, pixel_id])**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6

def test_dstress_dphase_2_quad_pts(plot=False):
    ### ----- Set up ----- ###
    # Discretization
    nb_grid_pts = [5, 7]
    dim = len(nb_grid_pts)
    lengths = [2.5, 3.1]
    formulation = µ.Formulation.small_strain

    gradient = [muFFT.Stencils2D.d_10_00, muFFT.Stencils2D.d_01_00,
                muFFT.Stencils2D.d_11_01, muFFT.Stencils2D.d_11_10]
    weights=[1, 1]

    # Material
    phase = np.random.random(nb_grid_pts).flatten(order='F')
    delta_Young = 10.
    delta_Poisson = 0.3
    phase[0] = 0

    # Load cases
    DelFs = [np.zeros([dim, dim]), np.zeros([dim, dim])]
    DelFs[0][0, 0] = 0.01
    DelFs[1][0, 1] = 0.007 / 2
    DelFs[1][1, 0] = 0.007 / 2

    # muSpectre solver parameters
    tol = 1e-6
    maxiter = 100
    verbose = µ.Verbosity.Silent

    # List of finite differences
    if plot:
        delta_list = [1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7, 1e-7]
    else:
        delta_list = [1e-4, 5e-5]


    ### ----- Analytical derivation ----- ###
    cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights=weights)
    # Material initialization
    Young = delta_Young * phase
    Poisson = delta_Poisson * phase
    mat = µ.material.MaterialLinearElastic4_2d.make(cell, "material")
    for pixel_id in cell.pixel_indices:
        mat.add_pixel(pixel_id, Young[pixel_id], Poisson[pixel_id])

    # muSpectre calculation
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    solver=µ.solvers.KrylovSolverCG(cell, tol, maxiter, verbose)
    stresses = []
    strains = []
    for DelF in DelFs:
        r = µ.solvers.newton_cg(cell, DelF, solver, tol, tol, verbose)
        stress = r.stress.copy().flatten(order='F')
        stresses.append(stress)
        strain = r.grad.copy().reshape(shape, order='F')
        strains.append(strain)

    # Derivative
    derivs = sa.calculate_dstress_dphase(cell, strains, Young, delta_Young,
                                         Poisson, delta_Poisson, gradient=gradient, weights=weights)

    ### ----- Finite difference derivation ----- ###
    shape = [dim, dim, cell.nb_quad_pts, *cell.nb_subdomain_grid_pts]
    shape2 = [dim, dim, cell.nb_quad_pts, cell.nb_pixels]
    diff_list = []
    for delta in delta_list:
        diff = 0
        for i_case, strain in enumerate(strains):
            deriv = derivs[i_case].reshape(shape2)
            stress = stresses[i_case].reshape(deriv.shape, order='F')
            deriv_fin_diff = np.empty(deriv.shape)

            #for i in range(len(phase)):
            for pixel_id in cell.pixel_indices:
                # New material
                helper_cell = µ.Cell(nb_grid_pts, lengths, formulation, gradient, weights=weights)
                mat = µ.material.MaterialLinearElastic4_2d.make(helper_cell, "material")
                for iter_pixel, pixel in cell.pixels.enumerate():
                    if pixel_id == iter_pixel:
                        mat.add_pixel(iter_pixel, Young[pixel_id] + delta_Young * delta,
                                      Poisson[pixel_id] + delta_Poisson * delta)
                    else:
                        mat.add_pixel(iter_pixel, Young[pixel_id], Poisson[pixel_id])

                # Derivative
                stress_plus = helper_cell.evaluate_stress(strain.reshape(shape, order='F'))
                stress_plus = stress_plus.reshape(deriv.shape, order='F')
                deriv_fin_diff[:, :, :, pixel_id] = (stress_plus[:, :, :, pixel_id] - stress[:, :, :, pixel_id]) / delta

            diff += np.linalg.norm(deriv[:, :, :, pixel_id] - deriv_fin_diff[:, :, :, pixel_id])**2
        diff = np.sqrt(diff)
        diff_list.append(diff)

    ### ----- Exponential fit ----- ###
    alpha = np.log(diff_list[0] + 1) / delta_list[0]

    ### ----- Plotting (optional) ----- ###
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('Fin. diff.')
        ax.set_ylabel('Norm of difference of square error derivative with respect to strains')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.plot(delta_list, diff_list, marker='o', label='Calculated')
        delta_list = np.array(delta_list)
        ax.plot(delta_list, np.exp(alpha * delta_list) - 1, '--', marker='x', label='Exp-fit')
        ax.legend()
        plt.show()

    assert abs((np.exp(alpha * delta_list[1]) - 1) - diff_list[1]) <= 1e-6


