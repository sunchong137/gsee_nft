import numpy as np
from scipy.special import eval_chebyt
from scipy import integrate


def dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_order, nmesh=100):
    """
    Evaluate the Fourier coefficients the approximate heaviside function.
    Args:
        max_dft_order       : int, cutoff of the DFT expansion (d in the paper).
        rescaled_energy_acc : float, energy accuracy * tau (delta in the paper).  
        dft_order           : int or array of int, the order(s) at which the coeffcients are evaluated (j in the paper)
        nmesh               : number of points to construct the smear Dirac function.
    """
    dft_smear_dirac_k = dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_order, nmesh)
    dft_heaviside_k = dft_coeffs_heaviside(dft_order)
    dft_approx_heaviside_k = np.sqrt(2 * np.pi) * dft_smear_dirac_k * dft_heaviside_k
    return dft_approx_heaviside_k


def approx_heaviside_from_dft(d, delt, x):
    """
    Evaluate the approximate Heaviside at x.
    """
    k = np.arange(-d, d + 1)
    Fk = dft_coeffs_approx_heaviside(d, delt, k, nmesh=200)
    lk = 2 * d + 1
    lx = len(x)
    F = np.dot(Fk, np.exp(1.0j * np.kron(k, x).reshape(lk, lx))) / np.sqrt(2 * np.pi)

    return F


def approx_heaviside_from_convol(d, delt, nmesh=200):
    x = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    M = eval_smear_dirac(d, delt, nmesh)
    H = _gen_heaviside(nmesh)
    F = np.convolve(M, H, mode="same")
    # F = signal.fftconvolve(M, H, mode="same")
    return F
    

def eval_smear_dirac(d, delt, nmesh, thr=1e-5):
    """
    Evaluate smear Dirac delta function in Lemma 5 at [-pi, pi].
    Args:
        d: degree of Chebyshev polynomials
        delt: parameter
        nmesh: number of mesh points (assume uniform grid for now)
    Returns:
        1d array of the function values.
    """
    assert abs(delt - np.pi) > thr and abs(delt + np.pi) > thr  # no overflow!
    x = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    x_n = 1 + 2 * (np.cos(x) - np.cos(delt)) / (1 + np.cos(delt))
    M = eval_chebyt(d, x_n)
    M /= integrate.simpson(M, x)

    return M


def eval_dft_coeffs_slow(y, k, x=None):
    """
    Slow version of discrete Fourier transform.
    Args:
        x: original space points in [-pi, pi]
        y: f(x)
        k: reciprical space point(s), 1d array
    Returns:
        Fourier coefficients of f(x) at k.
    """
    lx = len(y)
    if x is None:
        x = np.linspace(-np.pi, np.pi, lx)
    try:  # k is provided as an array
        lk = len(k)
        # lx = len(x)
        n = np.exp(-1.0j * np.einsum("i, j -> ij", k, x))
        n = np.einsum("i, ji -> ji", y, n)
        # x_n = np.kron(np.ones(lk), x).reshape(lk, lx)
        yk = integrate.simpson(n, x) / np.sqrt(2 * np.pi)
        # print("k is an array!")

    except:
        # print("k is given as a number!")
        n = y * np.exp(-1.0j * k * x)
        yk = integrate.simpson(n, x) / np.sqrt(2 * np.pi)

    return yk


def dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_order, nmesh, thr=1e-5):
    """
    Fourier transform of the smear dirac function.
    """
    smear_dirac_x = eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh, thr=thr)
    dft_smear_dirac_k = eval_dft_coeffs_slow(smear_dirac_x, dft_order)
    return dft_smear_dirac_k


def dft_coeffs_heaviside(dft_order):
    """
    FFT of the periodic Heaviside function.
    Args:
        dft_order: int or array of int.
    Returns:
        complex number or array of complex number.
    """
    try:
        np.seterr(divide="ignore", invalid="ignore")
        num_order = len(dft_order)
        dft_heaviside_k = np.zeros(num_order)
        dft_heaviside_k = -2.0j * (dft_order % 2) / (np.sqrt(2 * np.pi) * dft_order)  # complain when k = 0
        dft_heaviside_k[np.where(dft_order == 0)] = np.sqrt(np.pi / 2.0)

    except:
        if dft_order == 0:
            dft_heaviside_k = np.sqrt(np.pi / 2.0)
        elif dft_order % 2 == 0:
            dft_heaviside_k = 0
        else:
            dft_heaviside_k = -2.0j / (np.sqrt(2 * np.pi) * dft_order)

    return dft_heaviside_k


def rescale_hamiltonian_spectrum(hamiltonian, bound=np.pi/3):
    """
    Rescaling the hamiltonian, returns the rescaling factor tau.
    Suppose we can diagonalize the Hamiltonian.
    Args:
        hamiltonian : 2D array, the matrix representation of the Hamiltonian
        bound       : the targeted upper limit of the spectrum of the Hamiltonian
    Returns:
        Float (tau in the paper).
    """
    energies, _ = np.linalg.eigh(hamiltonian)
    energy_rescaling_factor = bound / max(abs(energies[0]), abs(energies[-1]))

    return energy_rescaling_factor


def _gen_heaviside(nmesh):
    """
    Return the Heaviside function in an array of size (nmesh + 1).
    """
    heaviside_array = np.ones(nmesh + 1)
    heaviside_array[: int(nmesh // 2)] *= 0

    return heaviside_array
