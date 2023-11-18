# torch_qip
The repository includes the the implemented circuits in qiskit, the efficient state simulator torch_qip, and experiments conducted in the paper.

files on circuit implementation:
- ./circuits_qiskit/main_simulate.py: the implemented circuit simulator
- ./circuits_qiskit/main_real_machine.py: run the circuit on IBM real quantum machine

files on the efficient state simulator:
- ./src: the source code of the efficient state simulator

files related to experiments:
- ./circuits_qiskit/compare_circuit_torchqip.py: compare the results of circuit simulator in qiskit and the proposed efficient state simulator
- ./evaluations/: codes to evaluate the efficiency and accuracy of the state simulator
- ./examples: the examples of using our torch_qip module in machine learning and deep learning

The following instructions are how to use our efficient state simulator in practifcal use:

# Compilation and Installation

To install, run
```
python setup.py install
```

To check whether the intallation is successful, run
```
sh test.sh
```

# Usage

```
import torch # must import torch before torch_qip
import torch_qip
```
