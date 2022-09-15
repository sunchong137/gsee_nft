import numpy as np
from scipy.special import eval_chebyt
from scipy import integrate


def dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_grid, nmesh=100):
    """
    Evaluate the Fourier coefficients the approximate heaviside function.
    Args:
        max_dft_order       : int, cutoff of the DFT expansion (d in the paper).
        rescaled_energy_acc : float, energy accuracy * tau (delta in the paper).  
        dft_grid           : int or array of int, the order(s) at which the coeffcients are evaluated (j in the paper)
        nmesh               : number of points to construct the smear Dirac function.
    """
    dft_smear_dirac_k = dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grid, nmesh)
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
    smear_dirac = eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh)
    heaviside = _gen_heaviside(nmesh)
    approx_heaviside = np.convolve(smear_dirac, heaviside, mode="same")

    return approx_heaviside
    

def eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh, thr=1e-5):
    """
    Evaluate smear Dirac delta function in Lemma 5 at [-pi, pi].
    Args:
        max_dft_order: degree of Chebyshev polynomials
        rescaled_energy_acc: parameter
        nmesh: number of mesh points (assume uniform grid for now)
    Returns:
        1d array of the function values.
    """
    assert abs(rescaled_energy_acc - np.pi) > thr and abs(rescaled_energy_acc + np.pi) > thr  # no overflow!
    energy_grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    energy_grids_n = 1 + 2 * (np.cos(energy_grids) - np.cos(rescaled_energy_acc)) / (1 + np.cos(rescaled_energy_acc))
    smear_dirac = eval_chebyt(max_dft_order, energy_grids_n)
    smear_dirac /= integrate.simpson(smear_dirac, energy_grids)

    return smear_dirac


def eval_dft_coeffs_slow(func, dft_grid, grid=None):
    """
    Slow version of discrete Fourier transform.
    Args:
        func : the function to be transformed
        grid : original space points in [-pi, pi]
        ft_order : reciprical space point(s), 1d array
    Returns:
        Fourier coefficients of f(x) at k.
    """
    len_grid = len(func)
    if grid is None:
        grid = np.linspace(-np.pi, np.pi, len_grid)
    try:  # k is provided as an array
        # len_dft_grid = len(dft_grid)
        integrand = np.exp(-1.0j * np.einsum("i, j -> ij", dft_grid, grid))
        integrand = np.einsum("i, ji -> ji", func, integrand)
        dft_func = integrate.simpson(integrand, grid) / np.sqrt(2 * np.pi)

    except:
        integrand = func * np.exp(-1.0j * dft_grid * grid)
        dft_func = integrate.simpson(integrand, grid) / np.sqrt(2 * np.pi)

    return dft_func


def dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grid, nmesh, thr=1e-5):
    """
    Fourier transform of the smear dirac function.
    """
    smear_dirac_x = eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh, thr=thr)
    dft_smear_dirac_k = eval_dft_coeffs_slow(smear_dirac_x, dft_grid)
    return dft_smear_dirac_k


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
