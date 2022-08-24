import numpy as np
import helpers
import one_qubit_circ

pi = np.pi 


def measure_Xj_1q(input_state_vector, hamiltonian, j_val, energy_rescalor=None):
    '''
    Measure the real part of Tr[\rho exp(-i j tau H)]
    One qubit case.
    Args:
        input_state_vector: vector, initial state
        hamiltonian: matrix, Hamiltonian
        j_val: int, parameter sampled
        energy_rescalor: float, rescaling factor of the Hamiltonian (tau)
    returns:
        An int number to be either 1 or -1.
    '''
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_slow(hamiltonian)

    full_state_vector = one_qubit_circ.main_circuit_1q(input_state_vector, hamiltonian, energy_rescalor, j_val, id="X")
    ancilla_output = one_qubit_circ.measure_ancilla(full_state_vector)
    Xj = -1. * (2 * ancilla_output - 1) # 0 -> 1, 1 -> -1
    return Xj

def measure_Yj_1q(input_state_vector, hamiltonian, j_val, energy_rescalor=None):
    '''
    Measure the imaginary part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    '''
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_slow(hamiltonian)

    full_state_vector = one_qubit_circ.main_circuit_1q(input_state_vector, hamiltonian, energy_rescalor, j_val, id="Y")
    ancilla_output = one_qubit_circ.measure_ancilla(full_state_vector)
    Yj = -1. * (2 * ancilla_output - 1) # 0 -> 1, 1 -> -1
    return Yj
    
def eval_acdf_single_sample(energy_grids, j_val, Zj, angle_j):
    '''
    Evaluate the G_function at points stored in x - energy grids.
    G = F_tot * (Xj + iYj) * exp[i (ang_j + j * x)]
    Args:
        energy_grids - points at which the G function is evaluated.
        F_tot - sum of the norms of DFT coeffs of the approximate Heaviside function.
        j_val - int 
        Zj -  a complex number that can be (+- 1 +- 1j)
        angle_j - a number (angle)
    Returns:
        1D array of the size of x.
    '''
    G = Zj * np.exp(1.j * (angle_j + j_val * energy_grids))
    return G

def classical_sampler(num_samples, max_dft_order, rescaled_energy_acc, nmesh=200):
    '''
    Generate the j values from [-max_dft_order, max_dft_order]
    Args:
        num_samples -  number of samples
        max_dft_order - cutoff of the DFT expansion (d in the paper)
        rescaled_energy_acc - energy accuracy * tau (delta in the paper)  
        nmesh - number of grids to generate a mesh of real space points
    Returns:  
        Fj - an array of complex numbers (DFT of the approximate Heaviside function) 
        j_samp - an array of integers of size num_samples.
    '''

    # generate Fj Pr(k = j) = |Fj| / \sum_j |Fj|
    j_all = np.arange(-max_dft_order, max_dft_order+1, 1)
    Fj = helpers.dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, j_all, nmesh)
    Fj_abs = np.abs(Fj)

    # evaluate probability distribution Pr(J = j) = |Fj| / \sum_j |Fj|
    F_tot = np.sum(Fj_abs)
    pr_j = Fj_abs / F_tot

    # generate random j values 
    j_samp = np.random.choice(j_all, size=num_samples, p=pr_j)

    return Fj, j_samp


def quantum_sampler(j_samp, input_state, hamiltonian, energy_rescalor=None):
    #TODO write a test
    '''
    Sampling J and XJ + iYJ.
    Args:
        j_samp: a sample of integers in [-max_dft_order, max_dft_order], generated by classical sampling
        input_state: initial state
        hamiltonian: Hamiltonian
        energy_rescalor: rescaling factor of the Hamiltonian (tau in the paper)
    Returns:
        Zs - array of complex numbers, set of (XJ + iYJ) values.
    '''

    # generate Xj and Yj values
    X_samp = []
    Y_samp = []
    for j in j_samp:
        X_samp.append(measure_Xj_1q(input_state, hamiltonian, j, energy_rescalor))
        Y_samp.append(measure_Yj_1q(input_state, hamiltonian, j, energy_rescalor))

    X_samp = np.asarray(X_samp)
    Y_samp = np.asarray(Y_samp)
    Z_samp = X_samp + 1.j * Y_samp

    return Z_samp

def adcf_kernel(max_dft_order, Fj, j_samp, Z_samp, energy_grid=None, nmesh=200):
    '''
    Evaluate ACDF given Fj, j_samp, Z_samp
    Args:
        max_dft_order - cutoff of the DFT expansion (d in the paper)
        Fj - Fourier transforms of the approximate Heaviside function
        j_samp - a sample of integers in [-max_dft_order, max_dft_order], generated by classical sampling
        Z_samp - a sample of complex numbers in {+-1 +- 1.j}, generated by the quantum computer
        energy_grid - points where the function is evaluated
        nmesh - number of grids to generate a mesh of real space points
    Returns:
        An array containing the values of the approximate CDF at points in energy_grid.
    '''
    # TODO rewrite the test

    num_samples = len(j_samp)
    Fj_angle = np.angle(Fj)

    # generate G
    if energy_grid is None:
        energy_grid = np.linspace(-pi, pi, nmesh, endpoint=True)
    num_grid = energy_grid.shape[-1]
    G_samp = np.zeros(num_grid, dtype=np.complex)

    for i in range(num_samples):
        j, Zj = j_samp[i], Z_samp[i]
        ang_j = Fj_angle[j + max_dft_order] 
        G = eval_acdf_single_sample(energy_grid, j, Zj, ang_j)
        G_samp += G

    Fj_abs = np.abs(Fj)
    F_tot = np.sum(Fj_abs) 

    G_samp = G_samp * F_tot / num_samples

    return G_samp