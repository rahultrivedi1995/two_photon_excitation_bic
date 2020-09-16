"""FDTD for two-photon scattering from time-delayed feedback systems with
multiple emitters."""
import numpy as np

from typing import List, Tuple, Optional


class TwoPhFdtdGrid:
    def __init__(
            self,
            num_states: int,
            max_time: float,
            tau_lims: Tuple[float, float],
            dt: float,
            init_state: np.ndarray) -> None:
        """Creates a new `TwoPhFdtdGrid` object.

        Args:
            num_states: The number of states to maintain during the FDTD.
            max_time: The maximum simulation time.
            tau_lims: The limits of the positions (in units of time) that the
                localized systems interact with.
            dt: The time-step to use in the FDTD. This is used to discretize
                both the times and the time-bins.
            init_state: The initial two photon state that excites the
                localized systems.
        """
        self._num_states = num_states
        self._dt = dt
        # Calculate the number of time-steps within the simulation time and the
        # interaction range for the localized systems.
        self._num_tsteps = int(max_time // dt)
        self._num_int_tbins = int((tau_lims[1] - tau_lims[0]) // dt) + 1
        self._min_tau = np.min(tau_lims)
        # While saving the maximum tau, we use the number of time bins and the
        # the discretization to perform the computation to account for
        # rounding error that might arise due to discretization.
        self._max_tau = self._min_tau + self._dt * (self._num_int_tbins - 1)
        # Number of time-bins that will be populated during FDTD.
        self._num_tbins = self._num_tsteps + self._num_int_tbins

        # Setup the FDTD state vectors.
        self._state = [np.zeros((self._num_tbins, self._num_tsteps),
                                dtype=complex) for _ in range(self._num_states)]
        self._ex_state = np.zeros((self._num_states,
                                   self._num_states,
                                   self._num_tsteps), dtype=complex)

        # Save the initial two-photon state. The initial two-photon state is
        # assumed to start from the time-bin after the time bin corresponding
        # to the last time step. Furthermore, if the temporal extent of the
        # incident two-photon state is larger than the simulation time, then
        # we only save the data corresponding to the simulation time.
        if init_state.shape[0] > self._num_tsteps:
            self._init_state = init_state[0:self._num_tsteps,
                                          0:self._num_tsteps]
        else:
            self._init_state = init_state

    @property
    def num_tsteps(self) -> int:
        return self._num_tsteps

    @property
    def num_tbins(self) -> int:
        return self._num_tbins

    @property
    def num_int_tbins(self) -> int:
        return self._num_int_tbins

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def state(self) -> List[np.ndarray]:
        return self._state

    @property
    def ex_state(self) -> np.ndarray:
        return self._ex_state

    def time_to_ind(self, t: float) -> int:
        return int(t // self._dt)

    def tau_to_ind(self, tau: float) -> int:
        return int((tau - self._min_tau) // self._dt)

    def get_ph_state_val(
            self,
            index: int,
            time: float,
            tau: float) -> complex:
        """Returns the value of a state at a particular position and time."""
        time_ind = self.time_to_ind(time)
        tau_ind = self.tau_to_ind(tau)
        return self._state[index][tau_ind, time_ind]

    def get_ex_state_val(
            self,
            indices: Tuple[int, int],
            time: float) -> complex:
        """Returns the value of the state at a particular time."""
        time_ind = self.time_to_ind(time)
        return self._state[indices[0], indices[1], time_ind]

    def get_time_slice(
            self,
            index: int,
            time: float,
            tau_extent: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Calculates a time slice.

        For a state `psi(tau, t; n)`, a time slice is defined by
        `psi(tau_1:tau_2, t; n)` if `t > 0` and 0 if `t <= 0`.

        Args:
            index: The state to be accessed.
            time: The time `t` at which to take the time-slice.
            tau_extent: The extents `tau_1` to `tau_2` to use in taking slice.
                If this not specified, then it is assumed to be the entire
                `tau` axis.

        Returns:
            The time-slice as an array.
        """
        # Calculate the indices corresponding to `tau_extent`.
        min_tau_ind = (0 if tau_extent is None
                       else self.tau_to_ind(tau_extent[0]))
        max_tau_ind = (self._num_tbins - 1 if tau_extent is None
                       else self.tau_to_ind(tau_extent[1]))
        # If time is smaller than 0, then return 0.
        if time < 0:
            return np.zeros(self._num_tbins, dtype=complex)

        # Calculate the time index.
        time_ind = self.time_to_ind(time)
        if time_ind == 0:
            return 0.5 * self._state[index][min_tau_ind:(max_tau_ind + 1), 0]
        else:
            return 0.5 * (
                    self._state[index][min_tau_ind:(max_tau_ind + 1), time_ind] +
                    self._state[index][min_tau_ind:(max_tau_ind + 1), time_ind - 1])

    def get_tau_transpose_slice(
            self,
            index: int,
            time: float,
            tau_extent: Tuple[float, float]) -> np.ndarray:
        """Calculate a transposed slice for a given time-bin.

        For a state `psi(tau, t; n)`, the tau-transpose slice is given by
        `psi(t, tau; n) * Theta(tau_1 < tau < tau_2)`.

        Args:
            index: The state which to slice.
            time: The time instance at which to calculate slice.
            tau_extent: The extent of the time-bins with which to slice.

        Returns:
            The slice as a numpy array.
        """
        # Calculate the time index.
        time_ind = self.tau_to_ind(time)
        # Calculate the indices for tau.
        min_tau_ind = self.tau_to_ind(tau_extent[0])
        max_tau_ind = self.tau_to_ind(tau_extent[1]) - 1
        num_tau_ind = max_tau_ind - min_tau_ind + 1
        # Calculate the slice vector.
        slice_vec = np.zeros(self._num_tbins, dtype=complex)
        slice_vec[(min_tau_ind):(max_tau_ind + 1)] = 0.5 * (
                self._state[index][time_ind, 0:num_tau_ind] +
                self._state[index][time_ind, 1:(num_tau_ind + 1)])
        return slice_vec

    def get_init_state_slice(
            self,
            time: float) -> np.ndarray:
        """Get the slice of initial state relevant for an update."""
        # Calculate the tau index corresponding to the time `t`.
        ind = self.tau_to_ind(time)
        slice_vec = np.zeros(self._num_tbins, dtype=complex)
        if (ind >= self.num_int_tbins and
            ind < self.num_int_tbins + self._init_state.shape[0]):
            min_ind = self.num_int_tbins
            max_ind = self.num_int_tbins + self._init_state.shape[0]
            slice_vec[min_ind:max_ind] = self._init_state[
                :, ind - self.num_int_tbins]
        return slice_vec


def _update_reabsorption(
        time: float,
        index: int,
        state: TwoPhFdtdGrid,
        delays: List[float],
        gammas: List[float],
        phases: List[complex]) -> np.ndarray:
    """Calculates the source in two-photon FDTD due to reabsorption.

    Args:
        time: The time at which to compute the reabsorption term.
        index: The index of the state which to update.
        state: The state of the two-photon wave-packet.
        delays: The delays of the two-level systems from the mirror.
        gammas: The decays corresponding to the coupling constants of the
            two-level system with the waveguide mode.
        phases: The mirror phases seen by the fields interacting with the
            different two-level system.

    Returns:
        The source corresponding to re-absorption term in two-photon FDTD.
    """
    # Calculate the update for the given index.
    update_vec = np.zeros(state.num_tbins, dtype=complex)
    for n in range(state.num_states):
        # There are four re-absorption terms that need to be accounted for.
        # We add them one by one.
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) *
                       state.get_tau_transpose_slice(
                           n, time + delays[index],
                           (delays[n], time + delays[n])))
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) * np.conj(phases[n]) *
                       state.get_tau_transpose_slice(
                           n, time + delays[index],
                           (-delays[n], time - delays[n])))
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) * phases[index] *
                       state.get_tau_transpose_slice(
                           n, time - delays[index],
                           (delays[n], time + delays[n])))
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) *
                       np.conj(phases[n]) * phases[index] *
                       state.get_tau_transpose_slice(
                           n, time - delays[index],
                           (-delays[n], time - delays[n])))

    return update_vec

def _update_feedback(
        time: float,
        index: int,
        state: TwoPhFdtdGrid,
        delays: List[float],
        gammas: List[float],
        phases: List[complex]) -> np.ndarray:
    """Calculates the source corresponding to the time-delayed feedback."""
    update_vec = np.zeros(state.num_tbins, dtype=complex)
    # This term has contribution from feedback from other emitters and
    # feedback via mirror.
    for n in range(index):
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) *
                       phases[index] * np.conj(phases[n]) *
                       state.get_time_slice(
                           n, time - delays[index] + delays[n]))

    for n in range(index + 1, state.num_states):
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) *
                       state.get_time_slice(
                           n, time - delays[n] + delays[index]))

    # Contributon via mirror.
    for n in range(state.num_states):
        update_vec -= (np.sqrt(gammas[index] * gammas[n]) *
                       phases[index] *
                       state.get_time_slice(
                           n, time - delays[index] - delays[n]))

    return update_vec


def _update_emission(
        time: float,
        index: int,
        state: TwoPhFdtdGrid,
        delays: List[float],
        gammas: List[float],
        phases: List[complex]) -> np.ndarray:
    """Calculates the emission from time-delayed feedback system.

    This update should be performed after the update for the excited state
    population.
    """
    update_vec = np.zeros(state.num_tbins, dtype=complex)
    for n in range(state.num_states):
        # Calculate the relevant excited state vector.
        time_ind = state.time_to_ind(time)
        #        ex_state = 0.5 * (state.ex_state[n, index, time_ind] +
        #                  state.ex_state[n, index, time_ind + 1])
        ex_state = state.ex_state[n, index, time_ind]
        # Direct emission contribution.
        tau_index = state.tau_to_ind(time + delays[n])
        update_vec[tau_index] -= 2.0j * ex_state * np.sqrt(gammas[n])
        # Contribution from time-delayed feedback.
        tau_index = state.tau_to_ind(time - delays[n])
        update_vec[tau_index] -= 2.0j * ex_state * np.sqrt(gammas[n]) * np.conj(phases[n])

    return update_vec



def _update_inc_fld(
        time: float,
        index: int,
        state: TwoPhFdtdGrid,
        delays: List[float],
        gammas: List[float],
        phases: List[complex]) -> np.ndarray:
    """Calculates the source corresponding to the incident field."""
    update_vec = np.zeros(state.num_tbins, dtype=complex)
    update_vec -= (2.0j * np.sqrt(gammas[index]) *
                   state.get_init_state_slice(time + delays[index]))
    update_vec -= (2.0j * np.sqrt(gammas[index]) * phases[index] *
                   state.get_init_state_slice(time - delays[index]))
    return update_vec


def _update_ex_state(
        time: float,
        indices: Tuple[int, int],
        state: TwoPhFdtdGrid,
        delays: List[float],
        gammas: List[float],
        phases: List[float]) -> np.ndarray:
    """Computes the udpdate for the two-photon pair under consideration."""
    update_val = -0.5j * (
            np.sqrt(gammas[indices[1]]) * state.get_ph_state_val(
                indices[0], time, time + delays[indices[1]]) +
            np.sqrt(gammas[indices[0]]) * state.get_ph_state_val(
                indices[1], time, time + delays[indices[0]]) +
            phases[indices[1]] * np.sqrt(gammas[indices[1]]) * state.get_ph_state_val(
                indices[0], time, time - delays[indices[1]]) +
            phases[indices[0]] * np.sqrt(gammas[indices[0]]) * state.get_ph_state_val(
                indices[1], time, time - delays[indices[0]]))
    return update_val


def simulate(
        delays: List[float],
        gammas: List[float],
        phis: List[float],
        sim_time: float,
        dt: float,
        two_ph_psi_init: np.ndarray) -> np.ndarray:
    """Simulate two-photon FDTD from multi-emitter time-delayed system."""
    # We compute the phases so that we are not computing exponentials at
    # every time step.
    phases = [np.exp(1.0j * phi) for phi in phis]
    # Make the FDTD grid object.
    state_obj = TwoPhFdtdGrid(len(delays), sim_time,
                              (-np.max(delays), np.max(delays)), dt,
                              two_ph_psi_init)
    for time_ind in range(1, state_obj.num_tsteps):
        # Update the excited state.
        for m in range(state_obj.num_states):
            for n in range(state_obj.num_states):
                if m != n:
                    src_vec = _update_ex_state(
                            (time_ind - 0.5) * dt, (m, n), state_obj,
                            delays, gammas, phases)
                    alpha = ((1.0 - 0.5 * (gammas[m] + gammas[n]) * dt) /
                             (1.0 + 0.5 * (gammas[m] + gammas[n]) * dt))
                    beta = dt / (1.0 + 0.5 * (gammas[m] + gammas[n]) * dt)
                    state_obj.ex_state[m, n, time_ind] = (
                            alpha * state_obj.ex_state[m, n, time_ind - 1] +
                            beta * src_vec)

        for index in range(state_obj.num_states):
            src_vec = (_update_reabsorption((time_ind - 0.5) * dt, index,
                                            state_obj, delays,
                                            gammas, phases) +
                       _update_feedback((time_ind - 0.5) * dt, index,
                                        state_obj, delays,
                                        gammas, phases) +
                       _update_emission((time_ind - 0.5) * dt, index,
                                        state_obj, delays,
                                        gammas, phases) / dt +
                       _update_inc_fld((time_ind - 0.5) * dt, index,
                                       state_obj, delays,
                                       gammas, phases))
            # Calculate the update coefficients.
            alpha = ((1.0 - 0.5 * gammas[index] * dt) /
                     (1.0 + 0.5 * gammas[index] * dt))
            beta = dt / (1.0 + 0.5 * gammas[index] * dt)
            state_obj.state[index][:, time_ind] = (
                    alpha * state_obj.state[index][:, time_ind - 1] +
                    beta * src_vec)


    # Return data.
    return state_obj.state, state_obj.ex_state

