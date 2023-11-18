import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import QFT
from qiskit.providers.fake_provider import FakeHanoi
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit_aer.noise import NoiseModel
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

import time

def get_initial_gate(target_state, label=None):
    k = int(np.ceil(np.log2(len(target_state))))
    assert len(target_state) == 2 ** k

    circ = QuantumCircuit(k)
    circ.initialize(target_state, range(k))
    gate = transpile(circ, basis_gates=['u', 'cx']).to_gate(label=label)
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
        control_gate = UnitaryGate(unitary_gate, label=f'$G^{{{2 ** index}}}$').control(1)
        cir.append(control_gate, [qubit, *input_reg])
        unitary_gate = unitary_gate.power(2)
    cir.append(QFT(accuracy_t, do_swaps=False, inverse=True).to_instruction(), reversed(ansatz))
    return cir


def QIP(vector_z, vector_x, t, shots=None):
    if shots is None: shots = 500*2**t

    norm = np.linalg.norm(vector_z) * np.linalg.norm(vector_x)

    vector_z = vector_z / np.linalg.norm(vector_z)
    vector_x = vector_x / np.linalg.norm(vector_x)
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
    qc.measure(range(k + 1, t + k + 1), reversed(range(t)))
    # Output drawn circuits
    decomposed_qc = qc.decompose()  # Does not modify original circuit
    decomposed_qc.draw('mpl', reverse_bits=True)
    plt.show()

    backend = FakeHanoi()
    job = backend.run(transpile(qc, backend, optimization_level=3), shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    # the results
    print(counts)
    plot_histogram(counts)
    plt.show()
    mode_R = int(max(counts, key=counts.get), 2)
    # print(R)
    print(max(counts, key=counts.get))
    # print(sum(dict(counts).values()))

    def R2result(R):
        return -np.cos(np.pi*R/2**(t-1)) * norm

    mode_result = R2result(mode_R)
    counts = dict(counts)
    avg_result = np.sum(
        np.array(list(counts.values())) / np.sum(list(counts.values())) * \
        np.array(list(map(
            lambda x: R2result(int(x, 2)), 
            counts.keys()))))
    return mode_result, avg_result

def eval_efficiency():
    x = torch.load("../x.pth")
    w = torch.load("../w.pth")

if __name__ == "__main__":
    # the dim of vectors should be 2^n, e.g. 2，4，8
    # vector1 = np.array([-np.sqrt(2), np.sqrt(3)])
    # vector2 = np.array([2, 3])
    # print("the accurate result:", np.inner(vector1, vector2))

    # # t controls the precision
    # result = QIP(vector1, vector2, 6, 100)
    # print("the QIP result:", result)



    vector1 = np.random.rand(4)
    vector2 = np.random.rand(4)
    print("the accurate result:", np.inner(vector1, vector2))

    # t controls the precision
    result = QIP(vector1, vector2, 6)
    print("the QIP result:", result)