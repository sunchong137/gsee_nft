import numpy as np
import helpers
import one_qubit_circ

pi = np.pi 

# TODO absorb tau into ham.

def measure_Xj_1q(state, ham, j, tau=None):
    '''
    Measure the real part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    Args:
        state: initial state
        ham: Hamiltonian
        j: parameter sampled
        tau: rescaling factor of the Hamiltonian
    returns:
        a number to be either 1 or -1.
    '''
    if tau is None:
        tau = helpers.rescale_ham_slow(ham)

    full_state = one_qubit_circ.main_circuit_1q(state, ham, tau, j, id="X")
    _Xj = one_qubit_circ.measure_ancilla(full_state)
    Xj = -1. * (2 * _Xj - 1) # 0 -> 1, 1 -> -1
    return Xj

def measure_Yj_1q(state, ham, j, tau=None):
    '''
    Measure the imaginary part of Tr[\rho exp(-i k tau H)]
    One qubit case.
    '''
    if tau is None:
        tau = helpers.rescale_ham_slow(ham)

    full_state = one_qubit_circ.main_circuit_1q(state, ham, tau, j, id="Y")
    _Yj = one_qubit_circ.measure_ancilla(full_state)
    Yj = -1. * (2 * _Yj - 1) # 0 -> 1, 1 -> -1
    return Yj
    
def eval_G(x, F_tot, j, Xj, Yj, ang_j):
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
    G = F_tot * (Xj + 1.j * Yj) * np.exp(1.j * (ang_j + j * x))
    return G

def adcf_sampler_1q(Ns, d, delt, state, ham, tau=None, x=None, nmesh=200):
    '''
    Evaluate ACDF by sampling J and XJ + iYJ.
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

    # generate Fj Pr(k = j) = |Fj| / \sum_j |Fj|
    j_all = np.arange(-d, d+1, 1)
    Fj = helpers.dft_aheaviside(d, delt, j_all, nmesh)
    Fj_abs = np.abs(Fj)
    Fj_angle = np.angle(Fj)

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

    # generate G
    if x is None:
        x = np.linspace(-pi, pi, nmesh, endpoint=True)
    len_x = x.shape[-1]
    G_samp = np.zeros(len_x, dtype=np.complex)

    for i in range(Ns):
        j, Xj, Yj = j_samp[i], X_samp[i], Y_samp[i]
        ang_j = Fj_angle[j + d] 
        G = eval_G(x, F_tot, j, Xj, Yj, ang_j)
        G_samp += G

    G_samp /= Ns

    return G_samp