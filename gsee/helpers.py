import numpy as np
from scipy.special import eval_chebyt
from scipy import integrate


def dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_grids, nmesh=2000):
    """
    Evaluate the Fourier coefficients the approximate heaviside function at dft_grids.
    Args:
        max_dft_order       : int, cutoff of the DFT expansion (d in the paper).
        rescaled_energy_acc : float, energy accuracy * tau (delta in the paper).  
        dft_grids           : int or array of int, the order(s) at which the coeffcients are evaluated (j in the paper)
        nmesh               : number of points to construct the smear Dirac function.
    Returns:
        Array or a float.
    """
    grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    dft_smear_dirac = _eval_dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grids, grids=grids, nmesh=nmesh)
    dft_heaviside = _eval_dft_coeffs_heaviside(dft_grids)
    dft_approx_heaviside_k = np.sqrt(2 * np.pi) * dft_smear_dirac * dft_heaviside
    
    return dft_approx_heaviside_k


#
# The following functions are private and only called in this file.
#
def _eval_smear_dirac(max_dft_order, rescaled_energy_acc, grids=None, nmesh=None, thr=1e-5):
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

def _eval_dft_coeffs_smear_dirac(max_dft_order, rescaled_energy_acc, dft_grids, grids=None, nmesh=None, thr=1e-5):
    """
    Fourier transform of the smear dirac function.
    """
    if grids is None:
        assert nmesh is not None
        grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
    smear_dirac = _eval_smear_dirac(max_dft_order, rescaled_energy_acc, grids=grids, thr=thr)
    dft_smear_dirac = _eval_dft_coeffs_slow(smear_dirac, dft_grids, grids)
    return dft_smear_dirac


def _eval_dft_coeffs_slow(vals_func, dft_grids, grids):
    """
    Slow version of discrete Fourier transform.
    Args:
        vals_func  : the array of the values of function f(x) to be transformed
        dft_grid   : the grid for the recipracal space
        grid       : original space points in [-pi, pi]
    Returns:
        Fourier coefficients of f(x) at dft_grid.
    NOTE: the number of grids has to be large enough (> 10^4) to reach decent accuracy.
    """
    
    try:  
        integrand = np.exp(-1.0j * np.einsum("i, j -> ij", dft_grids, grids))
        integrand = np.einsum("i, ji -> ji", vals_func, integrand)
        dft_func = integrate.simpson(integrand, grids) / np.sqrt(2 * np.pi)

    except:
        integrand = vals_func * np.exp(-1.0j * dft_grids * grids)
        dft_func = integrate.simpson(integrand, grids) / np.sqrt(2 * np.pi)

    return dft_func


def _eval_dft_coeffs_heaviside(dft_grids):
    """
    FFT of the periodic Heaviside function.
    Args:
        dft_grid: int or array of int.
    Returns:
        complex number or array of complex number.
    """
    try:
        np.seterr(divide="ignore", invalid="ignore")
        num_order = len(dft_grids)
        dft_heaviside_k = np.zeros(num_order)
        dft_heaviside_k = -2.0j * (dft_grids % 2) / (np.sqrt(2 * np.pi) * dft_grids)  # complain when k = 0
        dft_heaviside_k[np.where(dft_grids == 0)] = np.sqrt(np.pi / 2.0)

    except:
        if dft_grids == 0:
            dft_heaviside_k = np.sqrt(np.pi / 2.0)
        elif dft_grids % 2 == 0:
            dft_heaviside_k = 0
        else:
            dft_heaviside_k = -2.0j / (np.sqrt(2 * np.pi) * dft_grids)

    return dft_heaviside_k


def _gen_heaviside(grids=None, nmesh=None):
    """
    Return the Heaviside function in an array of size (nmesh + 1).
    """
    if grids is not None:
        nmesh = len(grids)
        heaviside_array = np.ones(nmesh)
        heaviside_array[np.where(grids < 0)] *= 0
    else:
        assert nmesh is not None
        heaviside_array = np.ones(nmesh + 1)
        heaviside_array[: int(nmesh // 2)] *= 0

    return heaviside_array

#
# The following two functions were not used anywhere, will keep them 
# until I am sure they will never be used.
#
# def _approx_heaviside_from_dft(max_dft_order, rescaled_energy_acc, energy_grids):
#     """
#     Evaluate the approximate Heaviside at energy grid points.
#     """
#     dft_grids = np.arange(-max_dft_order, max_dft_order + 1)
#     dft_heaviside_k = dft_coeffs_approx_heaviside(max_dft_order, rescaled_energy_acc, dft_grids, nmesh=200)
#     len_dft = 2 * max_dft_order + 1
#     len_energy = len(energy_grids)
#     approx_heaviside = np.dot(dft_heaviside_k, np.exp(1.0j * np.kron(dft_grids, energy_grids).reshape(len_dft, len_energy))) / np.sqrt(2 * np.pi)

#     return approx_heaviside


# def _approx_heaviside_from_convol(max_dft_order, rescaled_energy_acc, nmesh=2000):
#     '''
#     Approximate Heaviside function from convolution of smear dirac function and periodic heaviside function.
#     '''
#     grids = np.linspace(-np.pi, np.pi, nmesh + 1, endpoint=True)
#     smear_dirac = _eval_smear_dirac(max_dft_order, rescaled_energy_acc, nmesh=nmesh, grids=grids)
#     heaviside = _gen_heaviside(grids=grids, nmesh=nmesh)
#     approx_heaviside = np.convolve(smear_dirac, heaviside, mode="same")

#     return approx_heaviside
    