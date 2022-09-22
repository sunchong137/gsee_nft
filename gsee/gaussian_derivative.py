import numpy as np
from gsee import helpers
from gsee import quantum_circuits

def gaussian_derivative_classical_sampler(
    num_samples, max_dft_order, energy_gap, rescaled_energy_acc, nmesh=200
):
    """
    Generate the j values from [-max_dft_order, max_dft_order]
    Args:
        num_samples -  number of samples
        max_dft_order - cutoff of the DFT expansion (d in the paper)
        rescaled_energy_acc - energy accuracy * tau (delta in the paper)
        nmesh - number of grids to generate a mesh of real space points
    Returns:
        Fj - an array of complex numbers (DFT of the approximate gaussian_derivative function)
        j_samp - an array of integers of size num_samples.
    """

    # generate Fj Pr(k = j) = |Fj| / \sum_j |Fj|
    j_all = np.arange(-max_dft_order, max_dft_order + 1, 1)
    sigma = 0.02 * energy_gap
    # Fj = 2 * np.pi * (j_all) * (np.exp(-((1 / 2) * (np.pi * j_all * sigma) ** 2)))
    Fj = (
        2.0j
        * np.pi
        * j_all
        * np.exp(-((j_all * np.pi * sigma) ** 2))
        / (4 * len(j_all))
    )
    # Fj = helpers.dft_coeffs_approx_gaussian_derivative(
    #     max_dft_order, rescaled_energy_acc, j_all, nmesh
    # )
    Fj_abs = np.abs(Fj)

    # evaluate probability distribution Pr(J = j) = |Fj| / \sum_j |Fj|
    F_tot = np.sum(Fj_abs)
    pr_j = Fj_abs / F_tot

    # generate random j values
    j_samp = np.random.choice(j_all, size=num_samples, p=pr_j)

    return Fj, j_samp


def quantum_sampler(j_samp, input_state, hamiltonian, energy_rescalor=None):
    # TODO write a test
    """
    Sampling J and XJ + iYJ.
    Args:
        j_samp: a sample of integers in [-max_dft_order, max_dft_order], generated by classical sampling
        input_state: initial state
        hamiltonian: Hamiltonian
        energy_rescalor: rescaling factor of the Hamiltonian (tau in the paper)
    Returns:
        Zs - array of complex numbers, set of (XJ + iYJ) values.
    """

    # generate Xj and Yj values
    X_samp = []
    Y_samp = []
    for j in j_samp:
        X_samp.append(quantum_circuits.measure_Xj_1qubit(input_state, hamiltonian, j, energy_rescalor))
        Y_samp.append(quantum_circuits.measure_Yj_1qubit(input_state, hamiltonian, j, energy_rescalor))

    X_samp = np.asarray(X_samp)
    Y_samp = np.asarray(Y_samp)
    Z_samp = X_samp + 1.0j * Y_samp

    return Z_samp


def generate_convolution_estimate_from_sample(energy_grids, j_val, Zj, angle_j):
    """
    Evaluate the G_function at points stored in x - energy grids.
    G = F_tot * (Xj + iYj) * exp[i (ang_j + j * x)]
    Args:
        energy_grids - points at which the G function is evaluated.
        F_tot - sum of the norms of DFT coeffs of the approximate gaussian_derivative function.
        j_val - int
        Zj -  a complex number that can be (+- 1 +- 1j)
        angle_j - a number (angle)
    Returns:
        1D array of the size of x.
    """
    G = Zj * np.exp(1.0j * (angle_j + j_val * energy_grids))
    return G


def gaussian_derivative_kernel(
    max_dft_order, Fj, j_samp, Z_samp, energy_grid=None, nmesh=200
):
    """
    Evaluate Gaussian derivative kernel given Fj, j_samp, Z_samp
    Args:
        max_dft_order - cutoff of the DFT expansion (d in the paper)
        Fj - Fourier transforms of the approximate Gaussian derivative function
        j_samp - a sample of integers in [-max_dft_order, max_dft_order], generated by classical sampling
        Z_samp - a sample of complex numbers in {+-1 +- 1.j}, generated by the quantum computer
        energy_grid - points where the function is evaluated
        nmesh - number of grids to generate a mesh of real space points
    Returns:
        An array containing the values of the approximate Gaussian derivative at points in energy_grid.
    """
    # TODO rewrite the test

    num_samples = len(j_samp)
    Fj_angle = np.angle(Fj)
    # generate G
    if energy_grid is None:
        energy_grid = np.linspace(-np.pi, np.pi, nmesh, endpoint=True)
    try:
        num_grid = energy_grid.shape[-1]
        G_samp = np.zeros(num_grid, dtype=np.complex)
    except:
        num_grid = 1
        G_samp = 0.0

    for i in range(num_samples):
        j, Zj = j_samp[i], Z_samp[i]
        ang_j = Fj_angle[j + max_dft_order]
        G = generate_convolution_estimate_from_sample(energy_grid, j, Zj, ang_j)
        G_samp += G

    Fj_abs = np.abs(Fj)
    F_tot = np.sum(Fj_abs)

    G_samp = G_samp * F_tot / num_samples

    return G_samp
