# pip install qiskit[visualization]
# pip install qiskit-ibm-provider qiskit-ibm-runtime

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt
import pandas as pd

import time

def get_initial_gate(target_state, label=None):
    k = int(np.ceil(np.log2(len(target_state))))
    assert len(target_state) == 2 ** k

    circ = QuantumCircuit(k)
    circ.initialize(target_state, range(k), normalize=True)
    circ = transpile(circ, basis_gates=['u', 'cx'])
    del circ.data[0]
    gate = circ.to_gate(label=label)
    return gate


def get_unitary_gate(unitary_gate, k, label=None):
    circ = QuantumCircuit(k)
    gate = UnitaryGate(unitary_gate)
    circ.append(gate, range(k))
    circ = transpile(circ, basis_gates=['u', 'cx'])
    gate = circ.to_gate(label=label)
    return gate


def Modified_Hadamard_Test(qubit_num, vector_z, vector_x):
    input_reg = QuantumRegister(qubit_num)
    ansatz = QuantumRegister(1)
    cir = QuantumCircuit(input_reg, ansatz)
    cir.h(ansatz)

    z_control_gate = get_initial_gate(vector_z, '$U_z$').control(num_ctrl_qubits=1, ctrl_state=0)
    cir.append(z_control_gate, [ansatz, *input_reg])

    x_control_gate = get_initial_gate(vector_x, '$U_x$').control(num_ctrl_qubits=1)
    cir.append(x_control_gate, [ansatz, *input_reg])

    cir.h(ansatz)
    return cir


def QPE(qubit_num, unitary_gate, accuracy_t):
    input_reg = QuantumRegister(qubit_num)
    ansatz = QuantumRegister(accuracy_t)
    cir = QuantumCircuit(input_reg, ansatz)

    cir.h(ansatz)
    for index, qubit in enumerate(ansatz):
        control_gate = get_unitary_gate(unitary_gate, qubit_num, label=f'$G^{{{2 ** index}}}$').control(1)
        cir.append(control_gate, [qubit, *input_reg])
        unitary_gate = unitary_gate.power(2)
    cir.append(QFT(accuracy_t, do_swaps=False, inverse=True).to_instruction(), reversed(ansatz))
    return cir


def QIP(vector_z, vector_x, t):
    k = int(np.ceil(np.log2(len(vector_z))))
    assert len(vector_z) == 2 ** k

    modified_hadamard_test_circ = Modified_Hadamard_Test(k, vector_z, vector_x)

    vector_g = Statevector.from_int(0, 2 ** (k + 1)).evolve(modified_hadamard_test_circ)
    unitary_G = (Operator(Pauli((1 + k) * 'I')) - 2 * vector_g.to_operator()) @ Operator(Pauli('Z' + k * 'I'))
    qpe_circ = QPE(k + 1, unitary_G, t)

    qc = QuantumCircuit(t + k + 1, t)
    qc.append(modified_hadamard_test_circ.to_instruction(), range(k + 1))
    qc.barrier()
    qc.append(qpe_circ.to_instruction(), range(t + k + 1))
    qc.measure(range(k + 1, t + k + 1), range(t - 1, -1, -1))
    # circuit figures
    # decomposed_qc = qc.decompose()  # Does not modify original circuit
    # decomposed_qc.draw('mpl', reverse_bits=True, style="clifford")
    # plt.show()

    # save circuits
    # qasm = qc.decompose().qasm(filename='circuit.qasm')

    # load circuits
    # qc = QuantumCircuit.from_qasm_file("circuit.qasm")

    return qc


def run(model='simulate'):
    global norms, circuits, accuracies
    service = QiskitRuntimeService(
        channel="ibm_quantum")
    backend = service.get_backend("ibmq_qasm_simulator") if model == 'simulate' else service.get_backend('ibmq_brisbane')
    options = Options(optimization_level=3, resilience_level=1)
    jobs = []
    with Session(service=service, backend=backend):
        sample = Sampler(options=options)
        start_idx = 0
        while start_idx < len(circuits):
            end_idx = start_idx + min(backend.max_circuits, len(circuits) - start_idx)
            jobs.append(sample.run(circuits[start_idx:end_idx], shots=10000))
            start_idx = end_idx

    results = [job.result() for job in jobs]
    counts = [item for result in results for item in result.quasi_dists]
    Rs = [max(count, key=count.get) for count in counts]
    return [-np.cos(np.pi * R / 2 ** (t - 1)) * norm for R, norm, t in zip(Rs, norms, accuracies)]


if __name__ == "__main__":
    circuits = []
    norms = []
    # # dim of vectors should be 2^n，e.g. 2，4，8
    vectors_1 = [np.array([1, 1]), np.array([1, 1])]
    vectors_2 = [np.array([np.sqrt(2), np.sqrt(3)]), np.array([-1, 1])]
    accuracies = [4, 4]

    accurate_result = []
    for vector1, vector2 in zip(vectors_1, vectors_2):
        accurate_result.append(np.inner(vector1, vector2))
    print("accurate result:", accurate_result)

    for vector1, vector2, accuracy in zip(vectors_1, vectors_2, accuracies):
        circuits.append(QIP(vector1, vector2, accuracy))
        norms.append(np.linalg.norm(vector1) * np.linalg.norm(vector2))
    qip_simulate_result = run('simulate')
    print("the QIP simulate result:", qip_simulate_result)
    qip_real_result = run('real')
    print("the QIP real result:", qip_real_result)

    df = pd.DataFrame({'vectors_1': vectors_1, 'vectors_2': vectors_2,
                       'accuracy_t': accuracies, 'accurate result:': accurate_result,
                       'simulate result': qip_simulate_result, 'real result': qip_real_result})
    df.to_csv('result.csv', index=False)
    print("successfully saved the result to the result.csv.")
