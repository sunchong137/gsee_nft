import sys
sys.path.append('../')
from helpers import *
from matplotlib import pyplot as plt
from scipy.special import eval_chebyt
#from scipy import fft
import time

# The following tests are not unit tests, they are more intuitive...
# Unit tests will be added soon
def test_eval_chebyshev_slow():
    x = np.linspace(-1,1,200)
    dmax = 11
    trec = np.zeros(dmax)
    tsci = np.zeros(dmax)
    ds = np.arange(dmax)
    diff = np.zeros(dmax)
    for d in range(dmax):
        _t0 = time.time()
        t = eval_chebyshev_slow(x, d)
        _t1 = time.time()
        ts = eval_chebyt(d, x)
        _t2 = time.time()
        trec[d] = _t1 - _t0
        tsci[d] = _t2 - _t1
        print("time used for recursion: {}".format(_t1 - _t0))
        print("time used for scipy: {}".format(_t2 - _t1))
        diff[d] = np.linalg.norm(t-ts)
        #print(t.shape)
        #plt.plot(x, t)
    plt.plot(ds, diff)
    plt.show()

def test_eval_smear_dirac():
    nmesh = 200
    d = 20
    delt = 0.2
    x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
    M = eval_smear_dirac(d, delt, nmesh)
    d2 = 40
    M2 = eval_smear_dirac(d2, delt, nmesh)
    plt.plot(x, M, c='g', label=r"$d=20$")
    plt.plot(x, M2, '--', c='r', label=r"$d=40$")
    plt.legend()
    plt.xlabel(r"$x$")
    plt.ylabel(r"$M_{d, \delta}$")
    #plt.savefig("figures/smear_dirac.png", dpi=300)

def test_eval_dft_coeffs_slow():
    nmesh = 10
    d = 20
    delt = 0.2
    x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
    M = eval_smear_dirac(d, delt, nmesh)
    k = np.arange(-nmesh,nmesh+1,1)
    k = x.copy()
    Mk = eval_dft_coeffs_slow(M, k)
    #print(Mk)
    Mk2 = np.zeros(k.shape[0], dtype=np.complex128)
    for i in range(len(k)):
        Mk2[i] = eval_dft_coeffs_slow(M, k[i])
    #print(Mk2)
    #print(np.linalg.norm(Mk - Mk2))
    # Mks = fft.fftshift(M)/np.sqrt(2*np.pi)
    # plt.plot(x, Mks.real, label="scipy")
    # plt.plot(k, Mk.real, label="slow")
    # plt.legend()
    # plt.show()

def test_dft_coeffs_smear_dirac():
    nmesh = 10
    d = 20
    delt = 0.2
    k = np.arange(-nmesh,nmesh+1,1)
    Mk = dft_coeffs_smear_dirac(d, delt, k, nmesh)
    print(Mk)

def test_dft_coeffs_heaviside():
    k = np.arange(-4, 5)
    Hk = dft_coeffs_heaviside(k)
    print(Hk)
    Hks = []
    for i in k:
        Hks.append(dft_coeffs_heaviside(i))
    Hks = np.asarray(Hks)
    print(np.linalg.norm(Hks - Hk))
    print(k)

def test_dft_coeffs_approx_heaviside():
    nmesh = 10
    d = 20
    delt = 0.2
    k = np.arange(-nmesh,nmesh+1,1)
    Fk = dft_coeffs_approx_heaviside(d, delt, k, nmesh)
    print(Fk)
    

def test_approx_heaviside_from_dft():
    nmesh = 40
    d = 20
    delt = 0.2
    x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
    F = approx_heaviside_from_dft(d, delt, x)
    print(F)
    plt.plot(x, F)
    plt.show()

def test_approx_heaviside_from_convol():
    nmesh = 40
    d = 20
    delt = 0.2
    x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
    F = approx_heaviside_from_convol(d, delt, nmesh)
    plt.plot(x, F)
    plt.show()

def compare_aheaviside():

    nmesh = 80
    d = 40
    delt = 0.2
    x = np.linspace(-np.pi, np.pi, nmesh+1, endpoint=True)
    F1 = approx_heaviside_from_convol(d, delt, nmesh)*(2*np.pi/(nmesh+1))
    F2 = approx_heaviside_from_dft(d, delt, x)
    plt.plot(x, F1, label="convol")
    plt.plot(x, F2, '--', label="FT")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$F_{d, \delta}$")
    plt.legend()
    #plt.savefig("figures/aheaviside_compare.png", dpi=300)

def test_rescale_hamiltonian_slow():
    ham = np.random.rand(4,4)
    tau = rescale_hamiltonian_slow(ham)
    print(tau)

if __name__ == "__main__":
    test_eval_chebyshev_slow()
    test_eval_smear_dirac()
    test_eval_dft_coeffs_slow()
    test_dft_coeffs_smear_dirac()
    test_dft_coeffs_heaviside()
    test_dft_coeffs_approx_heaviside()
    test_approx_heaviside_from_dft()
    test_approx_heaviside_from_convol()
    compare_aheaviside()
    test_rescale_hamiltonian_slow()