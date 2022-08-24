import numpy as np
import helpers
import one_qubit_circ

pi = np.pi 


def measure_Xj_1q(input_state_vector, hamiltonian, j, energy_rescalor=None):
    '''
    Measure the real part of Tr[\rho exp(-i j tau H)]
    One qubit case.
    Args:
        input_state_vector: vector, initial state
        hamiltonian: matrix, Hamiltonian
        j: int, parameter sampled
        energy_rescalor: float, rescaling factor of the Hamiltonian (tau)
    returns:
        An int number to be either 1 or -1.
    '''
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_slow(hamiltonian)

    full_state_vector = one_qubit_circ.main_circuit_1q(input_state_vector, hamiltonian, energy_rescalor, j, id="X")
    ancilla_output = one_qubit_circ.measure_ancilla(full_state_vector)
    Xj = -1. * (2 * ancilla_output - 1) # 0 -> 1, 1 -> -1
    return Xj

def measure_Yj_1q(input_state_vector, hamiltonian, j, energy_rescalor=None):
    '''
    Measure the imaginary part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    '''
    if energy_rescalor is None:
        energy_rescalor = helpers.rescale_hamiltonian_slow(hamiltonian)

    full_state_vector = one_qubit_circ.main_circuit_1q(input_state_vector, hamiltonian, energy_rescalor, j, id="Y")
    ancilla_output = one_qubit_circ.measure_ancilla(full_state_vector)
    Yj = -1. * (2 * ancilla_output - 1) # 0 -> 1, 1 -> -1
    return Yj
    
def eval_G(x, j, Zj, ang_j):
    '''
    Evaluate the G_function at points stored in x.
    G = F_tot * (Xj + iYj) * exp[i (ang_j + j * x)]
    Args:
        x - points at which the G function is evaluated.
        F_tot - sum of the norms of DFT coeffs of the approximate Heaviside function.
        j - int 
        Xj - a number (+-1)
        Yj - a number (+-1)
        ang_j - a number (angle)
    Returns:
        1D array of the size of x.
    '''
    G = Zj * np.exp(1.j * (ang_j + j * x))
    return G

def sampler(Ns, d, delt, state, ham, tau=None, nmesh=200):
    #TODO write a test
    '''
    Sampling J and XJ + iYJ.
    Args:
        Ns -  number of samples
        d - cutoff
        delt - shift
        state: initial state
        ham: Hamiltonian
        tau: rescaling factor of the Hamiltonian
        nmesh - number of grids to generate a mesh of real space points
    Returns:
        Js, Zs - two arrays.
                 Js - array of int, a set of J values.
                 Zs - array of complex numbers, set of (XJ + iYJ) values.
    '''
    # generate Fj Pr(k = j) = |Fj| / \sum_j |Fj|
    j_all = np.arange(-d, d+1, 1)
    Fj = helpers.dft_coeffs_approx_heaviside(d, delt, j_all, nmesh)
    Fj_abs = np.abs(Fj)

    # evaluate probability distribution Pr(J = j) = |Fj| / \sum_j |Fj|
    F_tot = np.sum(Fj_abs)
    pr_j = Fj_abs / F_tot

    # generate random j values 
    j_samp = np.random.choice(j_all, size=Ns, p=pr_j)

    # generate Xj and Yj values
    X_samp = []
    Y_samp = []
    for j in j_samp:
        X_samp.append(measure_Xj_1q(state, ham, j, tau))
        Y_samp.append(measure_Yj_1q(state, ham, j, tau))

    X_samp = np.asarray(X_samp)
    Y_samp = np.asarray(Y_samp)
    Z_samp = X_samp + 1.j * Y_samp

    return Fj, j_samp, Z_samp

def adcf_kernel_1q(d, Fj, j_samp, Z_samp, x=None, nmesh=200):
    '''
    Evaluate ACDF given Fj, j_samp, Z_samp
    Args:
        Ns -  number of samples
        d - cutoff
        delt - shift
        state: initial state
        ham: Hamiltonian
        tau: rescaling factor of the Hamiltonian
        x - points where the function is evaluated
        nmesh - number of grids to generate a mesh of real space points
    Returns:
        Js, Zs - two arrays.
                 Js - array of int, a set of J values.
                 Zs - array of complex numbers, set of (XJ + iYJ) values.
    '''
    # TODO rewrite the test

    Ns = len(j_samp)
    Fj_angle = np.angle(Fj)

    # generate G
    if x is None:
        x = np.linspace(-pi, pi, nmesh, endpoint=True)
    len_x = x.shape[-1]
    G_samp = np.zeros(len_x, dtype=np.complex)

    for i in range(Ns):
        j, Zj = j_samp[i], Z_samp[i]
        ang_j = Fj_angle[j + d] 
        G = eval_G(x, j, Zj, ang_j)
        G_samp += G

    Fj_abs = np.abs(Fj)
    F_tot = np.sum(Fj_abs) # recalculating this ...

    G_samp = G_samp * F_tot / Ns

    return G_samp