"""Module for non Markovian QED system with bound states."""
import abc
import numpy as np

from typing import List


class BicSystem(metaclass=abc.ABCMeta):
    """Class to define an interface for a BIC system."""
    def __init__(self, num_qubits: int) -> None:
        self._num_qubits = num_qubits

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the system."""
        return self._num_qubits

    @abc.abstractmethod
    def compute_bound_state_olap(self) -> np.ndarray:
        """Computes the overlap of the bound states with the qubits."""
        raise NotImplementedError()

    @abc.abstractmethod
    def compute_scattering_state_olap(self, freqs: np.ndarray) -> np.ndarray:
        """Computes the overlap of the scattering states with the qubits."""
        raise NotImplementedError()


class TimeDelayedFeedbackSystem(BicSystem):
    """Class to define a time-delayed feedback system."""
    def __init__(
            self,
            delays: List[float],
            gammas: List[float]) -> None:
        """Creates a new `TimeDelayedFeedbackSystem` object.

        This system comprises of qubits placed in front of a PEC mirror. It is
        assumed that the mirror provides a phase of `pi` to all the reflected
        field. It is also assumed that all the qubits are at the same
        frequency.

        Args:
            delays: Delays of the qubits from the mirror.
            gammas: Coupling strength of the emitters to the waveguide modes.
                These are the individual coupling strengths.
        """
        if len(delays) != len(gammas):
            raise ValueError("Inconsistent number of qubits.")

        super(TimeDelayedFeedbackSystem, self).__init__(len(delays))
        self._delays = np.array(delays)
        self._gammas = np.array(gammas)

    def compute_bound_state_olap(self) -> np.ndarray:
        """Computes the bound state overlap."""
        # We construct the inner product matrix for photonic part of the
        # bound state.
        ph_to_coeff = np.zeros((2 * self.num_qubits - 1, self.num_qubits))
        delay_vec = np.zeros(2 * self.num_qubits - 1)
        # TODO(@rtrivedi): There is probably a more efficient way of
        # constructing this matrix, but since the number of qubits is likely
        # small, we just for loop.
        for n in range(self.num_qubits):
            ph_to_coeff[(self.num_qubits - 1 - n):
                        (self.num_qubits + n), n] = np.sqrt(self._gammas[n])
            if n == 0:
                delay_vec[self.num_qubits - 1] = 2 * self._delays[0]
            else:
                delay_vec[self.num_qubits - 1 - n] = (
                        self._delays[n] - self._delays[n - 1])
                delay_vec[self.num_qubits - 1 + n] = (
                        self._delays[n] - self._delays[n - 1])

        inner_prod_mat = np.eye(self.num_qubits) + ph_to_coeff.T @ (
                delay_vec[:, np.newaxis] * ph_to_coeff)
        # Construct the independent orthogonal bound states. If `W` is the
        # inner product matrix, then we have to construct a matrix `X` with
        # columns being the bound state overlap such that
        #               X.T * W * X = I
        # We do this by simply choosing X = sqrt(L^-1) U, where L is the
        # eigenvalue matrix and U is the eigenvector matrix of W.
        eig_vals, eig_vecs = np.linalg.eig(inner_prod_mat)
        alpha_mat = eig_vecs / np.sqrt(eig_vals)[np.newaxis, :]

        # Finally we construct the bound state overlap as transpose of the
        # alpha matrix.
        return alpha_mat

    def compute_scattering_state_olap(self, freqs: np.ndarray) -> np.ndarray:
        """Computes the overlap of the scattering states with the qubits."""
        # We will have to handle 0 frequency case separately. We calculate the
        # index where frequence is 0.
        zero_ind = np.where(freqs == 0)[0]
        # These overlaps can be computed by solving a system of equations per
        # frequency.
        # Calculate the source vector for each frequency.
        src_vec = (np.exp(1.0j * freqs[:, np.newaxis] *
                                 self._delays[np.newaxis, :]) -
                   np.exp(-1.0j * freqs[:, np.newaxis] *
                                  self._delays[np.newaxis, :])) / np.sqrt(
                                        2 * np.pi)
        # The zero index case has the source vector going to 0. So we replace
        # the source vector with its derivative with frequency.
        if zero_ind.size != 0:
            src_vec[zero_ind[0], :] = 2.0j * self._delays / np.sqrt(2 * np.pi)

        # Calculate the matrix for each frequency.
        mat = np.zeros((freqs.size, self.num_qubits, self.num_qubits),
                       dtype=complex)
        # TODO(@rtrivedi): There is probably a more efficient way of
        # constructing this matrix, but since the number of qubits is likely
        # small, we just for loop.
        for n in range(self.num_qubits):
            # Set the diagonal values.
            mat[:, n, n] = (
                    1.0j * self._gammas[n] *
                    (1 - np.exp(-2.0j * freqs * self._delays[n])) - freqs)
            # Set the off-diagonal values.
            mat[:, n, :n] = (
                    np.sqrt(self._gammas[n] * self._gammas[:n]) *
                    np.exp(-1.0j * freqs[:, np.newaxis] * self._delays[n]) *
                    (-2.0 * np.sin(freqs[:, np.newaxis] *
                                    self._delays[np.newaxis, :n])))
            mat[:, n, (n + 1):] = (
                    np.sqrt(self._gammas[n] * self._gammas[(n + 1):]) *
                    (-2.0 * np.sin(freqs[:, np.newaxis] * self._delays[n])) *
                    np.exp(-1.0j * freqs[:, np.newaxis] *
                           self._delays[np.newaxis, (n + 1):]))

        # Handle the zero vector case by using derivative of the matrix instead.
        if zero_ind.size != 0:
            for n in range(self.num_qubits):
                mat[zero_ind[0], n, n] = -(
                        1 + 2 * self._gammas[n] * self._delays[n])
                mat[zero_ind[0], n, :n] = (
                        np.sqrt(self._gammas[n] * self._gammas[:n]) *
                        (-2.0 * self._delays[:n]))
                mat[zero_ind[0], n, (n + 1):] = (
                        np.sqrt(self._gammas[n] * self._gammas[(n + 1):]) *
                        (-2.0 * self._delays[n]))

        # Finally, compute the overlap vectors. Note that this needs to be
        # conjugated to get the final overlap.
        xi_conj = np.linalg.solve(mat, src_vec)
        return np.conj(xi_conj)


