import sys
sys.path.append("../")
from acdf import *
import numpy as np
from matplotlib import pyplot as plt

pi = np.pi

def test_measure_Xj_1q():
    # generate a random Hamiltonian
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    # generate a random initial state
    state = np.random.rand(2)
    state /= np.linalg.norm(state)
    j = 2
    Ns = 100
    X = []
    for i in range(Ns):
        X.append(measure_Xj_1q(state, ham, j))
    print(X)

def test_measure_Yj_1q():
    # generate a random Hamiltonian
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    # generate a random initial state
    state = np.random.rand(2)
    state /= np.linalg.norm(state)
    j = 2
    Ns = 100
    Y = []
    for i in range(Ns):
        Y.append(measure_Yj_1q(state, ham, j))
    print(Y)

def test_eval_G():
    x = np.linspace(-pi/3, pi/3, 100)
    F_tot = 2.0
    j = 2
    Xj = 1.0
    Yj = -1.0
    ang_j = 0.1
    G = eval_G(x, F_tot, j, Xj, Yj, ang_j)
    print(G.shape)

def test_adcf_sampler_1q():
    Ns = 10000
    d = 40
    delt = 0.2
    nmesh = 500
    max_x = pi/2
    x = np.linspace(-max_x, max_x, nmesh)
    # generate a random Hamiltonian
    ham = np.random.rand(2, 2)
    ham = 0.5 * (ham + ham.T)
    ew, _ = np.linalg.eigh(ham)
    print("eigenvalues - E1: {}; E2: {}".format(ew[0], ew[1]))
    # generate a random initial state
    state = np.random.rand(2)
    state /= np.linalg.norm(state)
    G_bar = adcf_sampler_1q(Ns, d, delt, state, ham, x=x)

    plt.plot(x, G_bar)
    plt.xticks(ticks=ew, labels=["E0", "E1"])
    plt.xlabel(r"$E$")
    plt.ylabel(r"$\bar{G}$")
    plt.savefig("figures/G_bar.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    #test_gen_prob_Jk()
    #test_measure_Xj_1q()
    #test_measure_Yj_1q()
    #test_eval_G()
    test_adcf_sampler_1q()