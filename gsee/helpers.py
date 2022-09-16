import numpy as np
from scipy.special import eval_chebyt
from scipy import integrate


def dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_grid, nmesh=100):
    """
    Evaluate the Fourier coefficients the approximate heaviside function.
    Args:
        max_dft_order       : int, cutoff of the DFT expansion (d in the paper).
        rescaled_energy_acc : float, energy accuracy * tau (delta in the paper).  
        dft_grid            : int or array of int, the order(s) at which the coeffcients are evaluated (j in the paper)
        nmesh               : number of points to construct the smear Dirac function.
    """
    dft_smear_dirac_k = _eval_dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grid, nmesh)
    dft_heaviside_k = dft_coeffs_heaviside(dft_grid)
    dft_approx_heaviside_k = np.sqrt(2 * np.pi) * dft_smear_dirac_k * dft_heaviside_k
    return dft_approx_heaviside_k


def approx_heaviside_from_dft(max_dft_order, rescaled_energy_acc, energy_grids):
    """
    Evaluate the approximate Heaviside at energy grid points.
    """
    dft_grids = np.arange(-max_dft_order, max_dft_order + 1)
    dft_heaviside_k = dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_grids, nmesh=200)
    len_dft = 2 * max_dft_order + 1
    len_energy = len(energy_grids)
    approx_heaviside = np.dot(dft_heaviside_k, np.exp(1.0j * np.kron(dft_grids, energy_grids).reshape(len_dft, len_energy))) / np.sqrt(2 * np.pi)

    return approx_heaviside


def approx_heaviside_from_convol(max_dft_order, rescaled_energy_acc, nmesh=200):
    '''
    Approximate Heaviside function from 
    '''
    # x = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    smear_dirac = _eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh=nmesh, grids=None)
    heaviside = _gen_heaviside(nmesh)
    approx_heaviside = np.convolve(smear_dirac, heaviside, mode="same")

    return approx_heaviside
    

def _eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh=None, grids=None, thr=1e-5):
    """
    Evaluate smear Dirac delta function in Lemma 5 at [-pi, pi].
    Args:
        max_dft_order          : degree of Chebyshev polynomials
        rescaled_energy_acc    : the energy accuracy times the rescalor, delta in the paper.
        nmesh                  : number of mesh points (assume uniform grid for now)
    Returns:
        1d array of the function values.
    """
    assert abs(rescaled_energy_acc - np.pi) > thr and abs(rescaled_energy_acc + np.pi) > thr  # no overflow!
    if grids is None:    
        assert nmesh is not None
        grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
        
    grids_n = 1 + 2 * (np.cos(grids) - np.cos(rescaled_energy_acc)) / (1 + np.cos(rescaled_energy_acc))
    smear_dirac = eval_chebyt(max_dft_order, grids_n)
    smear_dirac /= integrate.simpson(smear_dirac, grids)

    return smear_dirac

def _eval_dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grids, nmesh, thr=1e-5):
    """
    Fourier transform of the smear dirac function.
    """
    grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    smear_dirac = _eval_smear_dirac(max_dft_order, rescaled_energy_acc, grids=grids, thr=thr)
    dft_smear_dirac = _eval_dft_coeffs_slow(smear_dirac, dft_grids, grids)
    return dft_smear_dirac


def _eval_dft_coeffs_slow(vals_func, dft_grids, grids):
    """
    Slow version of discrete Fourier transform.
    Args:
        vals_func       : the array of the values of function f(x) to be transformed
        dft_grid   : the grid for the recipracal space
        grid       : original space points in [-pi, pi]
    Returns:
        Fourier coefficients of f(x) at dft_grid.
    """
    
    try:  
        integrand = np.exp(-1.0j * np.einsum("i, j -> ij", dft_grids, grids))
        integrand = np.einsum("i, ji -> ji", vals_func, integrand)
        dft_func = integrate.simpson(integrand, grids) / np.sqrt(2 * np.pi)

    except:
        integrand = vals_func * np.exp(-1.0j * dft_grids * grids)
        dft_func = integrate.simpson(integrand, grids) / np.sqrt(2 * np.pi)

    return dft_func


def dft_coeffs_heaviside(dft_grid):
    """
    FFT of the periodic Heaviside function.
    Args:
        dft_grid: int or array of int.
    Returns:
        complex number or array of complex number.
    """
    try:
        np.seterr(divide="ignore", invalid="ignore")
        num_order = len(dft_grid)
        dft_heaviside_k = np.zeros(num_order)
        dft_heaviside_k = -2.0j * (dft_grid % 2) / (np.sqrt(2 * np.pi) * dft_grid)  # complain when k = 0
        dft_heaviside_k[np.where(dft_grid == 0)] = np.sqrt(np.pi / 2.0)

    except:
        if dft_grid == 0:
            dft_heaviside_k = np.sqrt(np.pi / 2.0)
        elif dft_grid % 2 == 0:
            dft_heaviside_k = 0
        else:
            dft_heaviside_k = -2.0j / (np.sqrt(2 * np.pi) * dft_grid)

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
    Return the Heaviside function in an array of s ize (nmesh + 1).
    """
    heaviside_array = np.ones(nmesh + 1)
    heaviside_array[: int(nmesh // 2)] *= 0

    return heaviside_array
