"""Calculating bound-state probability matrix."""
import numpy as np
import scipy.signal
from typing import List, Optional

import bic_system


def compute_cw_prob_matrix(
        freqs: np.ndarray,
        sys: bic_system.BicSystem,
        rep: Optional[float] = None) -> np.ndarray:
    """Compute the continuous-wave probability matrix."""
    # Compute some parameters regarding the frequency axis. It is assumed that
    # it is a uniformly spaced axis, extending symmetrically around 0 frequency
    # and having an odd number of frequency points.
    num_freqs = freqs.size
    num_pos_freqs = num_freqs // 2
    df = freqs[1] - freqs[0]

    # Compute the xi values and the probability functions.
    xis_mat = sys.compute_scattering_state_olap(freqs)
    eps = sys.compute_bound_state_olap()

    # Compute the decaying part of Green's function in time domain.
    times = (np.fft.fftfreq(num_freqs, df) * 2 * np.pi)[:num_pos_freqs]
    xi_sq_mat = np.matmul(xis_mat[:, :, np.newaxis],
                          xis_mat[:, np.newaxis, :].conj())
    xi_sq_t = (np.exp(1.0j * np.max(freqs) * times)[:, np.newaxis, np.newaxis] *
               np.fft.fft(xi_sq_mat, axis=0)[:num_pos_freqs, :, :] * df)
    eps_sq = eps @ eps.conj().T

    # Calculate the inverse of the Gamma field.
    Gamma_decay_part = np.zeros((num_freqs, sys.num_qubits, sys.num_qubits),
                                dtype=complex)
    print(xis_mat.shape)
    Gamma_decay_part[(num_pos_freqs + 1):, :, :] = (
        xi_sq_t**2 + 2 * eps_sq[np.newaxis, :, :] * xi_sq_t)
    phase_comp = np.exp(-2.0j * np.pi *
                        (num_pos_freqs + 1) / num_freqs * freqs / df)
    Gamma_decay_part_fft = (
        -2.0 * np.pi / df *
        phase_comp[:, np.newaxis, np.newaxis] *
        np.fft.ifftshift(np.fft.ifft(Gamma_decay_part, axis=0), axes=0))
    if rep is None:
        Gamma_inv = (freqs[:, np.newaxis, np.newaxis] * np.linalg.inv(
            1.0j * (eps_sq**2)[np.newaxis, :, :] +
            freqs[:, np.newaxis, np.newaxis] * Gamma_decay_part_fft))
    else:
        Gamma_inv = (freqs[:, np.newaxis, np.newaxis] * np.linalg.inv(
            1.0j * (eps_sq**2)[np.newaxis, :, :] +
            freqs[:, np.newaxis, np.newaxis] * (
                Gamma_decay_part_fft -
                1.0j / rep * np.eye(sys.num_qubits)[np.newaxis, :, :])))

    # Calculate convolution of xi with itself.
    xi_conv = scipy.signal.fftconvolve(
            xi_sq_mat, xi_sq_mat, "same", axes=0) * df
    # Finally calculate the probability matrix.
    out_state_olap = np.transpose(
        (eps.conj()[np.newaxis, :, :] * xis_mat.conj()[:, :, np.newaxis]),
        axes=[0, 2, 1])
    return (-2 * np.sqrt(2) * np.pi * (out_state_olap @ Gamma_inv @ xi_conv),
            xi_conv)


def compute_two_photon_state(
        sys: bic_system.BicSystem,
        coeffs: List[complex],
        cen_energy: float,
        tau: float,
        num_pts: int,
        dt: float) -> np.ndarray:
    """Calculate the energy entangled two-photon state.

    Args:
        sys: The system with the bound state.
        coeffs: The coefficients of the different scattering states.
        tau: The temporal spread of the energy wave-packet.
        num_pts: The number of points to use in constructing the two-photon
            wave-packet.
        dt: The time-step.

    Returns:
        The two-photon wavepacket.
    """
    # Calculate the frequencies at which to compute the wave-packet so as
    # align with the time-grid.
    half_num_pts = num_pts // 2
    df = 2 * np.pi / (num_pts * dt)
    freqs = np.arange(-half_num_pts, (num_pts - half_num_pts)) * df
    # Calculate the value of the scattering state overlaps.
    xi_vals = sys.compute_scattering_state_olap(freqs).T

    # Compute the energies.
    energies = freqs[:, np.newaxis] + freqs[np.newaxis, :]
    alpha_vals = np.exp(-0.5 * (energies - cen_energy)**2 * tau**2)
    # Construct the two-photon state in frequency domain.
    state_freq = 0
    for coeff, xi_val in zip(coeffs, xi_vals):
        state_freq += (coeff * alpha_vals *
                       np.conj(xi_val)[:, np.newaxis] *
                       np.conj(xi_val)[np.newaxis, :])

    # Calculate the time domain state.
    state_fft = np.fft.fft2(state_freq)
    phase_comp = np.exp(
        2.0j * np.pi * np.arange(num_pts) * half_num_pts / num_pts)
    state_time = np.fft.fftshift(
        (state_fft * phase_comp[:, np.newaxis]) * phase_comp[np.newaxis, :])
    state_time = state_time / np.sqrt(2 * np.sum(np.abs(state_time)**2 * dt**2))

    return state_time

