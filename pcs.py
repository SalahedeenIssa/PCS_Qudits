import cirq
import collections
from typing import Iterable

def create_noisy_circuit(circuit: cirq.Circuit, p:float=0.01, noisy_qubits:Iterable[cirq.Qid]=None):
    '''Creates a noisy circuit for Pauli Check Sandiwching
    that only applies errors to the U moments in the circuit,
    and keeps the checks noiseless.
    Args:
        circuit - The PCS circuit to make noisy
        p - The error probability to place on the qubits, default = 0.01
        noisy_qubits - The qubits that noise should be placed on. If not specified,
        the assumption is all logical qubits.
        ancilla_qubit - The qubit containing the ancilla. If not specified,
        the assumption is that the ancilla qubit is the last qubit in the
        circuit.
    Returns:
        A noisy PCS circuit, with errors applied to the U operations
    '''
    noise_model = cirq.NoiseModel.from_noise_model_like(cirq.depolarize(p))
    qubits = list(circuit.all_qubits())
    if noisy_qubits is None:
        noisy_qubits = qubits[:-1]
    noisy_moments = cirq.Circuit(noise_model.noisy_moments(circuit[2:-3], system_qubits=noisy_qubits))
    noisy_circuit = circuit[:2] + noisy_moments + circuit[-3:]
    return noisy_circuit

def simulate_noisy_circuit(noisy_circuit: cirq.Circuit, shots:int=1000):
    '''
    Function to simulate the noisy circuit on a cirq simulator, and return
    the results dictionary.
    Args:
        noisy_circuit - The noisy circuit to run
        shots - The number of shots. Default = 1000
    Returns:
        A dictionary containing the results counter
    '''
    result = cirq.sample(noisy_circuit, repetitions=shots)
    total_qubits = noisy_circuit.all_qubits()
    results_dict = result.multi_measurement_histogram(keys=[f'{qubit}' for qubit in total_qubits])
    return results_dict

def post_process_results(results_dict: collections.Counter, ancilla_index: int = -1):
    '''
    Post-processing the results to remove all the instances where the ancilla qubit was measured as a 1.
    Args:
        results_dict : The results dict of running the noisy simulation
        ancilla_index: The qubit from the qubit_list that is acting as the
        ancilla. Defaults to the last qubit.
    Returns:
        A filtered dictionary of the data qubits that removes all instances where the ancilla qubit
        was measured to be a 1.
    '''
    filtered_dictionary = {key[:-1]: value for key, value in results_dict.items() if key[-1] != 1}
    return filtered_dictionary





