import numpy as np
import acdf 

pi = np.pi

def bsearch_ground_state(N_samp, batches, eta, delt, d, Fj, j_samp, Z_samp, x_L=-pi/3, x_R=pi/3, nmesh=200):
    '''
    Given the acdf, search where the jump from zero to non-zero happens.
    Args:

    Returns:
    '''
    # TODO clean up Args.
    # TODO check if the two delt values are the same
    
    Ns_batch = int(N_samp / batches)
    if N_samp > Ns_batch * batches:
        Ns_batch += 1
        print("Adding {} samples to the ensamble.".format(Ns_batch * batches - N_samp))

    while x_L < x_R:

        x_M = (x_L + x_R) # take the medium point
        
        count_right = 0
        # generate batch samples
        for iter in batches:
            G_bar = acdf.adcf_kernel_1q(d, Fj, j_samp, Z_samp, x=x_M, nmesh=nmesh)
            if G_bar >= 0.75 * eta:
                count_right += 1
        
        # Shrink the area
        if count_right <= batches / 2:
            x_R = x_M + 2 * delt / 3.
        else:
            x_L = x_M - 2 * delt / 3.

    return (x_L + x_R) / 2.
            

def gen_global_values(acc_energy, tau, lower_ovlp, fail_prob):
    '''
    Generate values needed for the code.
    Args:
        acc_energy : required energy accuracy.
        tau        : rescaling factor of the Hamiltonian.
        lower_ovlp : the lower bound of overlap of init state and the ground state.
        fail_prop  : (1 - fail_prop) is the success probability.
    Returns:
        delt: float, rescaled energy accuracy
        d   : int, cutoff of the Fouries expansion
        batches : int, number of batches to fulfill majority voting
        N_samp : number of total samples
    '''

    delt = tau * acc_energy # rescaled energy accuracy
    d = int(1./delt * np.log(1./(delt * lower_ovlp))) # cutoff of Fourier expansion
    batches = np.log(1./fail_prob) + np.log(np.log(1./delt)) # number of batches to fulfill majority voting
    N_samp = 1. / lower_ovlp ** 2 * (np.log(d) ** 2) # number of total samples

    return delt, d, batches, N_samp
    
