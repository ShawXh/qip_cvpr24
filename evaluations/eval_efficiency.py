import torch, torch_qip
import numpy as np
import time
from torch.autograd.function import Function

class QIPMatMul(Function):
    @staticmethod
    def forward(ctx, input, weight, num_qubits, sample_times, out_mode):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight)
        output = torch_qip.qip(output, num_qubits, sample_times, out_mode)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = input.t().mm(grad_output)
        return grad_input, grad_weight, None, None, None

qipmm = QIPMatMul.apply

class QIPMatMulSlow(Function):
    @staticmethod
    def forward(ctx, input, weight, num_qubits, sample_times, out_mode):
        ctx.save_for_backward(input, weight)
        output = input.mm(weight)
        output = torch_qip.qip_all_case(output, num_qubits, sample_times, out_mode)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = input.t().mm(grad_output)
        return grad_input, grad_weight, None, None, None

qipmm_slow = QIPMatMulSlow.apply

x = torch.rand((16,128)) - 0.5
w = torch.rand((128,16)) - 0.5
x = x / torch.norm(x, p=2, dim=1, keepdim=True)
w = w / torch.norm(w, p=2, dim=0, keepdim=True)
x.requires_grad_(True)
w.requires_grad_(True)
loss = torch.norm(x.mm(w), p="fro")
loss.backward()


def run_classical(x, w):
    x.grad.zero_()
    w.grad.zero_()
    loss = torch.norm(x.mm(w), p="fro")
    loss.backward()

def run_quantum(x, w, num_qubits, sample_times, mode):
    x.grad.zero_()
    w.grad.zero_()
    loss = torch.norm(qipmm(x, w, num_qubits, sample_times, mode), p="fro")
    loss.backward()

def run_quantum_slow(x, w, num_qubits, sample_times, mode):
    x.grad.zero_()
    w.grad.zero_()
    loss = torch.norm(qipmm_slow(x, w, num_qubits, sample_times, mode), p="fro")
    loss.backward()


iters = 10000
num_quibts = [2, 4, 6 ,8]
# num_quibts = [6 ,8]
sample_times = [0, 1, 2, 3]
mode = ["mode"]
seed = list(range(10))

start_time = time.time()
for _ in range(iters):
    run_classical(x, w)
print("classical running time %.4fs" % (time.time() - start_time))

for n in num_quibts:
    for st in sample_times:
        for m in mode:
            print("=====Args: num_qubits={}, sample_times={}, mode={}".format(n, st, m))
            start_time = time.time()
            for _ in range(iters):
                run_quantum(x, w, n, st, m)
            print("quantum running time %.4fs" % (time.time() - start_time))

for n in num_quibts:
    for st in sample_times:
        for m in mode:
            print("=====Args: num_qubits={}, sample_times={}, mode={}".format(n, st, m))
            start_time = time.time()
            for _ in range(iters):
                run_quantum_slow(x, w, n, st, m)
            print("quantum running time %.4fs" % (time.time() - start_time))