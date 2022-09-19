from gsee.acdf import *
from gsee.helpers import *
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi



def test_eval_acdf_single_sample():
    x = np.linspace(-pi / 3, pi / 3, 100)
    F_tot = 2.0
    j = 2
    Xj = 1.0
    Yj = -1.0
    Zj = Xj + 1.0j * Yj
    ang_j = 0.1
    G = eval_acdf_single_sample(x, j, Zj, ang_j)
    print(G.shape)


def test_acdf_kernel():
    Ns = 10000
    d = 40
    delt = 0.2
    nmesh = 500
    max_x = pi / 2
    x = np.linspace(-max_x, max_x, nmesh)
    # generate a random Hamiltonian
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)

    # rescale hamiltonian
    tau = rescale_hamiltonian_spectrum(ham, bound=np.pi/3)
    ham *=  tau

    ew, ev = np.linalg.eigh(ham)
    state = ev[0] + np.random.rand(2) * 0.5  # make a good initial guess
    state /= np.linalg.norm(state)
    print("eigenvalues - E1: {}; E2: {}".format(ew[0], ew[1]))
    p0 = np.dot(ev[0].T, state) ** 2
    print("The overlap between the initial state and the ground state: {}".format(p0))
    # generate a random initial state
    Fj, j_samp = classical_sampler(Ns, d, delt, nmesh=nmesh)
    Z_samp = quantum_sampler(j_samp, state, ham, energy_rescalor=tau)

    G_bar = acdf_kernel(d, Fj, j_samp, Z_samp, energy_grid=x, nmesh=nmesh)

    plt.plot(x, G_bar)
    plt.xticks(ticks=ew, labels=["E0", "E1"])
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\bar{G}$")
    # plt.savefig("figures/G_bar.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    test_eval_acdf_single_sample()
    test_acdf_kernel()
