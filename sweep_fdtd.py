import numpy as np

import bic_system
import multi_bound_state_trapping as mbs_trapping
import two_photon_multi_emitter_fdtd as fdtd


def sweep_opt_trapping(tau: float):
    # Constant parameters over the sweep.
    dt = 0.01
    num_init_bins = 7000
    sim_time = 100

    # Delays to sweep over.
    delay_vals = np.linspace(0.01, 3.0, 25)
    bs_data = []
    for k, delay in enumerate(delay_vals):
        print("On delay = {}, done {}% of sweep".format(delay, k / delay_vals.size))
        # Make the system object.
        sys = bic_system.TimeDelayedFeedbackSystem([delay], [1.0])
        freqs = np.linspace(-1000, 1000, 200101) / delay
        prob_mat, xi_conv = mbs_trapping.compute_cw_prob_matrix(freqs, sys)
        prob_fun = np.abs(prob_mat[:, 0, 0])**2 / xi_conv[:, 0, 0]
        # Calculate the central frequency.
        cen_freq_ind = np.argmax(prob_fun)
        cen_freq = freqs[cen_freq_ind]
        # Setup the incident two-photon wave-packet.
        init_state = mbs_trapping.compute_two_photon_state(
                sys, [1], cen_freq, tau, num_init_bins, dt)

        # Perform the FDTD simulation
        state, _ = fdtd.simulate([delay], [1], [np.pi],
                                 sim_time, dt, init_state)
        bs_data.append(
                np.sum(np.abs(state[0])**2, axis=0) * dt * (1 + 2 * delay))
    np.save("results/opt_trapping_tau_{}.npy".format(tau), np.array(bs_data))



if __name__ == "__main__":
    print("Simulating tau = ", 1)
    sweep_opt_trapping(1)
    #print("Simulating tau = ", 2.5)
    #sweep_opt_trapping(2.5)
    #print("Simulating tau = ", 5)
    #sweep_opt_trapping(5)
    #print("Simulating tau = ", 10)
    #sweep_opt_trapping(10)



