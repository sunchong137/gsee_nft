import numpy as np
import acdf 

pi = np.pi

def bsearch_ground_state(Fj, j_samp, Z_samp, lower_bound_ovlp, energy_acc, energy_rescalor, fail_probability, x_L=-pi/3, x_R=pi/3, nmesh=200):
    '''
    Given the acdf, search where the jump from zero to non-zero happens.
    Args:
        tot_num_samples : total number of samples.
        num_batches : number of batches to perform the majority voting.
        lower_bound_ovlp : the lower bound of the overlap between the initial guess and the true ground state.

    Returns:
    '''

    rescaled_energy_acc, max_dft_order, num_batches, tot_num_samples = gen_global_values(energy_acc, energy_rescalor, lower_bound_ovlp, fail_probability)
    
    Ns_per_batch = int(tot_num_samples / num_batches)
    if tot_num_samples > Ns_per_batch * num_batches:
        Ns_per_batch += 1
        print("Adding {} samples to the ensamble.".format(Ns_per_batch * num_batches - tot_num_samples))

    while x_L < x_R:

        x_M = (x_L + x_R) # take the medium point
        
        count_right = 0
        # generate batch samples
        for iter in range(num_batches):
            G_bar = acdf.adcf_kernel(max_dft_order, Fj, j_samp, Z_samp, energy_grid=x_M, nmesh=nmesh)
            if G_bar >= 0.75 * lower_bound_ovlp:
                count_right += 1
        
        # Shrink the area
        if count_right <= num_batches / 2:
            x_R = x_M + 2 * rescaled_energy_acc / 3.
        else:
            x_L = x_M - 2 * rescaled_energy_acc / 3.

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
    batches = int(np.log(1./fail_prob) + np.log(np.log(1./delt))) # number of batches to fulfill majority voting
    N_samp = int(1. / lower_ovlp ** 2 * (np.log(d) ** 2)) # number of total samples

    return delt, d, batches, N_samp
    
