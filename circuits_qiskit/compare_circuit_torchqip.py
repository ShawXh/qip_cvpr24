import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector, Operator, Pauli
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import QFT
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakePerth, FakeHanoi
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session, Options
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt

import time

import torch, torch_qip
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import os
import sys


def get_initial_gate(target_state, label=None):
    k = int(np.ceil(np.log2(len(target_state))))
    assert len(target_state) == 2 ** k

    circ = QuantumCircuit(k)
    circ.initialize(target_state, range(k), normalize=True)
    circ = transpile(circ, basis_gates=['u', 'cx'])
    circ.data = [(instr, qargs, cargs) for instr, qargs, cargs in circ.data if instr.name != 'reset']
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

def QIP(vector_z, vector_x, t, model='simulate'):
    norm = np.linalg.norm(vector_z) * np.linalg.norm(vector_x)

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

    # results
    backend = AerSimulator() if model == 'simulate' else FakePerth()
    job = backend.run(transpile(qc, backend, optimization_level=3), shots=500 * 2 ** t)
    result = job.result()
    counts = result.get_counts(qc)
    # print("counts = ", counts)
    R = int(max(counts, key=counts.get), 2)

    def R2result(R):
        return -np.cos(np.pi*R/2**(t-1)) * norm
    mode_result = R2result(R)
    counts = dict(counts)
    avg_result = np.sum(
        np.array(list(counts.values())) / np.sum(list(counts.values())) * \
        np.array(list(map(
            lambda x: R2result(int(x, 2)),
            counts.keys()))))
    return mode_result, avg_result


if __name__ == "__main__":
    # the dim of vectors should be 2^n, e.g. 2，4，8

    N = 10
    d = 128
    t = 6
    sample_times = 500 * 2 ** t


    for d in [4, 16, 64]:
        if not os.path.exists("compare_circuit_torchqip_x_%dN_%dd.npy" % (N, d)):
            # x = np.array(torch.rand((N, d)) - 0.5)
            x = np.random.rand(N, d) - 0.5
            # print((x * x).sum(1, keepdims=True).shape)
            x /= np.sqrt((x * x).sum(1, keepdims=True))
            y = np.random.rand(N, d) - 0.5
            y /= np.sqrt((y * y).sum(1, keepdims=True))

            np.save("compare_circuit_torchqip_x_%dN_%dd.npy" % (N, d), x)
            np.save("compare_circuit_torchqip_y_%dN_%dd.npy" % (N, d), y)
        else:
            x = np.load("compare_circuit_torchqip_x_%dN_%dd.npy" % (N, d))
            y = np.load("compare_circuit_torchqip_y_%dN_%dd.npy" % (N, d))

        # vector1 = np.array([-np.sqrt(2), np.sqrt(3)])
        # vector2 = np.array([2, 3])
        # print("the accurate result:", np.inner(vector1, vector2))

        # # t controls the precision
        # result = QIP(vector1, vector2, t, sample_times)
        # print("the QIP result:", result)

        acc_res = np.sum(x * y, 1)
        # print("accurate results:", acc_res[:10])

        # torch_qip
        torch_qip.set_seed(0)
        start_time = time.time()
        mode_res = np.array(
            torch_qip.qip(torch.from_numpy(acc_res).clone(), t, sample_times, "mode")
        )
        end_time = time.time()
        # print(mode_res[:10])
        mse = mean_squared_error(acc_res, np.array(mode_res))
        # mse = ((acc_res - mode_res) * (acc_res - mode_res)).mean()
        mae = mean_absolute_error(acc_res, np.array(mode_res))
        # mae = np.abs(acc_res - mode_res).mean()
        print("Time: %.4f" % (end_time - start_time))
        print("MSE: %.8f, MAE: %.8f" % (mse, mae))

        os.system("cat /proc/{}/status | grep VmPeak".format(os.getpid()))

        # simulate circuit
        res = {"mode": [], "avg": []}
        start_time = time.time()
        for i in range(N):
            # print(x[i])
            mode_res, avg_res = QIP(x[i], y[i], t)
            res["mode"].append(mode_res)
            res["avg"].append(avg_res)
        end_time = time.time()
        acc_res = np.abs(acc_res)
        mode_res = np.abs(res["mode"])
        mse = mean_squared_error(np.abs(acc_res), mode_res)
        mae = mean_absolute_error(acc_res, mode_res)
        # print(res["mode"][:10])
        print("Time:", end_time - start_time)
        print("MSE:", mse, "MAE:", mae)
        os.system("cat /proc/{}/status | grep VmPeak".format(os.getpid()))
        

    
