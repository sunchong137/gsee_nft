import numpy as np
import acdf 

pi = np.pi

def search_ground_state(N_samp, batches, eta, delt, d, Fj, j_samp, Z_samp, x_L=-pi/3, x_R=pi/3, nmesh=200):
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
            
    
