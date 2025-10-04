import cirq
import numpy as np
import itertools
from typing import Sequence, Tuple


class QutritCZGate(cirq.Gate):
    '''
    An example gate to attempt to replicate the PCS circuit on a Qutrit. This is
    called a "CZ" gate, because it has the same impacts on the qubits |0> and |1>
    as a CZ gate does.
    '''
    def _qid_shape_(self):
        return (3,3)
    def _unitary_(self):
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, -1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, -1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1],])

    def _circuit_diagram_info_(self,args):
        return 'TCZ', 'TCZ'

class QutritZHalfGate(cirq.Gate):
    '''
    To reconcile with the fact that I have to create a lopsided set of
    Pauli checks, attempting to see what happens when I create Z^3/2,
    and use each one once
    '''
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return (1/np.sqrt(3))*np.array([[1,0,0],
                                        [0,omega**(3/2)]])

class QutritZGate(cirq.Gate):
    '''
    An example gate to attempt to replicate the PCS circuit on a Qutrit. This is
    called a "Z" gate, because it turns |0> to |0>, |1> to |-1>, and |2> to |-2>.
    '''
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]])

    def _circuit_diagram_info_(self, args):
        return 'TZ'

class QutritPlusGate(cirq.Gate):
    """A gate that adds one in the computational basis of a qutrit.

    This gate acts on three-level systems. In the computational basis of
    this system it enacts the transformation U|x〉 = |x + 1 mod 3〉, or
    in other words U|0〉 = |1〉, U|1〉 = |2〉, and U|2> = |0〉.
    """

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    def _circuit_diagram_info_(self, args):
        return '[+]'


class QutritPlusSquaredGate(cirq.Gate):
    '''
    Applies a plus gate squared
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,1,0],
                        [0,0,1],
                        [1,0,0]])

    def _circuit_diagram_info_(self, args):
        return '[++]'

class QutritChrestensonGate(cirq.Gate):
    '''
    An operator that achieves equal superposition among all the quantum
    basis states of a qutrit. Akin to a Hadamard for qubits.
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return (1/np.sqrt(3))*np.array([[1, 1, 1],
            [1, omega, omega**2],
            [1, omega**2, omega**4]])

    def _circuit_diagram_info_(self, args):
        return 'C3'

class ConjugateTransposeChrestenson(cirq.Gate):
    '''
    An operator that applies the conjugate transpose of the QutritChrestensonGate as a unitary.
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return (1/np.sqrt(3))*np.array([[1, 1, 1],
            [1, omega.conjugate(), (omega**2).conjugate()],
            [1, (omega**2).conjugate(), (omega**4).conjugate()]])

    def _circuit_diagram_info_(self, args):
        return 'CT3'


class ErrorY2Gate(cirq.Gate):
    '''
    An error gate that applies a "Y2" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,0,1],
                        [1,0,0],
                        [0,1,0]])

    def _circuit_diagram_info(self, args):
        return 'Y**2'


class ErrorYGate(cirq.Gate):
    '''
    An error gate that applies a "Y" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,1,0],
                 [0,0,1],
                 [1,0,0]])

    def _circuit_diagram_info(self, args):
        return 'Y'


class ErrorZGate(cirq.Gate):
    '''
    An error gate that applies a "Z" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[1,0,0],
                 [0,omega,0],
                 [0,0,omega**2]])

    def _circuit_diagram_info(self, args):
        return 'Z'


class ErrorZ2Gate(cirq.Gate):
    '''
    An error gate that applies a "Z squared" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[1,0,0],
                        [0, omega**2, 0],
                        [0,0,omega**4]])

    def _circuit_diagram_info(self, args):
        return 'Z**2'


class ErrorYZGate(cirq.Gate):
    '''
    An error gate that applies a "YZ" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0, omega, 0],
                        [0, 0, omega**2],
                        [1, 0, 0]])

    def _circuit_diagram_info(self, args):
        return 'YZ'


class ErrorY2ZGate(cirq.Gate):
    '''
    An error gate that applies a "YsquaredZ" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,0,omega**2],
                        [1,0,0],
                        [0,omega,0]])

    def _circuit_diagram_info(self, args):
        return 'Y2Z'


class ErrorYZ2Gate(cirq.Gate):
    '''
    An error gate that applies a "YZsquared" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,omega**2,0],
                        [0,0,omega**4],
                        [1,0,0]])

    def _circuit_diagram_info(self, args):
        return 'YZ2'


class ErrorY2Z2Gate(cirq.Gate):
    '''
    An error gate that applies a "YsquaredZsquared" error onto a given qutrit
    '''

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([[0,0,omega**4],
                        [1,0,0],
                        [0,omega**2,0]])

    def _circuit_diagram_info(self, args):
        return "Y2Z2"


class QuquartChrestensonGate(cirq.Gate):
    '''
    An operator that achieves equal superposition among all the quantum
    basis states of a ququart. Akin to a Hadamard for qubits.
    '''

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (4,)

    def _unitary_(self):
        return 0.5 * np.array(
            [[1, 1, 1, 1],
            [1, 1j, -1, -1j],
            [1, -1, 1, -1],
            [1, -1j, -1, 1j],
        ])

    def _circuit_diagram_info_(self, args):
        return 'C4'

class QutritIdentityGate(cirq.Gate):
    '''
    Identity gate for a Qutrit
    '''

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (3,)

    def _unitary_(self):
        return np.array(
            [[1,0,0],
             [0,1,0],
             [0,0,1],
             ]
        )

    def _circuit_diagram_info_(self, args):
        return 'I3'

class QutritZZGate(cirq.Gate):
    '''
    Tensor product of the QutritZ and the QutritZ gate, applying it to
    two qutrits.
    '''

    def _qid_shape_(self):
        return (3,3)

    def _unitary_(self):
        return np.array([
                 [1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, -1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, -1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, -1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0,],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, -1, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def _circuit_diagram_info(self, args):
        return 'TZZ', 'TZZ'



SHIFT = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]])
omega = np.exp(1j * 2 / 3 * np.pi)
CLOCK = np.array([[1, 0, 0],
                  [0, omega, 0],
                  [0, 0, omega ** 2]])



class QutritDepolarizingChannel(cirq.Gate):
    r"""A channel that depolarizes one qutrit.
    """

    def __init__(self, p: float) -> None:
        """Constructs a depolarization channel on a qutrit.

        Args:
            p: The probability that one of the shift/clock matrices is applied. Each of
                the 8 shift/clock gates is applied independently with probability
                $p / 8$.
            n_qubits: the number of qubits.

        Raises:
            ValueError: if p is not a valid probability.
        """

        error_probabilities = {}

        p_depol = p / 8
        p_identity = 1.0 - p
        for gate_pows in itertools.product(range(3), range(3)):
            if gate_pows == (0, 0):
                error_probabilities[gate_pows] = p_identity
            else:
                error_probabilities[gate_pows] = p_depol
        self.error_probabilities = error_probabilities
        self._p = p

    def _qid_shape_(self):
        return (3,)

    def _mixture_(self) -> Sequence[Tuple[float, np.ndarray]]:
        op = lambda shift_pow, clock_pow: np.linalg.matrix_power(SHIFT, shift_pow) @ np.linalg.matrix_power(CLOCK,
                                                                                                            clock_pow)
        return [(self.error_probabilities[(shift_pow, clock_pow)], op(shift_pow, clock_pow))
                for (shift_pow, clock_pow) in self.error_probabilities.keys()]

    def _has_mixture_(self) -> bool:
        return True

    def _value_equality_values_(self):
        return self._p

    def _circuit_diagram_info_(self, args):
        if args.precision is not None:
            return (f"D3({self._p:.{args.precision}g})",)
        else:
            return (f"D3({self._p})",)
        return result

    @property
    def p(self) -> float:
        """The probability that one of the qutrit gates is applied.

        Each of the 8 Pauli gates is applied independently with probability
        $p / 8$.
        """
        return self._p

    @property
    def n_qubits(self) -> int:
        """The number of qubits"""
        return 1
