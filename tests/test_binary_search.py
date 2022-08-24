from asyncio import base_tasks
import numpy
import sys
sys.path.append("../")
from binary_search import *
from acdf import *

def test_bsearch_ground_state():
    # set the parameters
    acc_energy = 0.01
    tau = 0.2
    lower_ovlp = 0.8
    fail_prob = 0.1

    delt, d, _, N_samp = gen_global_values(acc_energy, tau, lower_ovlp, fail_prob)

    nmesh = 500
    # generate a random Hamiltonian
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    
    ew, ev = np.linalg.eigh(ham)
    state = ev[0] + np.random.rand(2) * 0.5 # make a good initial guess
    state /= np.linalg.norm(state)
    print("eigenvalues - E1: {}; E2: {}".format(ew[0], ew[1]))
    p0 = np.dot(ev[0].T, state) ** 2
    print("The overlap between the initial state and the ground state: {}".format(p0))

    # sample
    Fj, j_samp = classical_sampler(N_samp, d, delt, nmesh=nmesh)
    Z_samp = quantum_sampler(j_samp, state, ham, energy_rescalor=tau)

    x_f = bsearch_ground_state(Fj, j_samp, Z_samp, lower_ovlp, acc_energy, tau, fail_prob, nmesh=nmesh)
    print("True grounds state energy: {}".format(ew[0]))
    print("Estimated ground state energy: {}".format(x_f))

def test_gen_global_values():
    acc_energy = 0.01
    tau = 0.2
    lower_ovlp = 0.8
    fail_prob = 0.1
    delt, d, batches, N_samp = gen_global_values(acc_energy, tau, lower_ovlp, fail_prob)
    print(delt, d, batches, N_samp)


if __name__ == "__main__":
    #test_gen_global_values()
    test_bsearch_ground_state()