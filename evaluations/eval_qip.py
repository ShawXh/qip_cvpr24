import torch, torch_qip
import numpy as np
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

data = np.load("./ip_data.npy", "r")
print("loading data...", data)
data = torch.from_numpy(data)



def run_epoch(num_qubits, sample_times, mode, seed):
    # print("=====Args: num_qubits={}, sample_times={}, mode={}, seed={}".format(num_qubits, sample_times, mode, seed))
    # print("ground truth data:", data)
    approximated_data = data.clone()
    start_time = time.time()
    torch_qip.set_seed(seed)
    torch_qip.qip(approximated_data, num_qubits, sample_times, mode)
    # print("used time %.2s" % (time.time() - start_time))

    # print("approximated data:", approximated_data)
    mse = mean_squared_error(data, approximated_data)
    mae = mean_absolute_error(data, approximated_data)

    # print("mse:", mse)
    # print("mae:", mae)
    return mse, mae

num_quibts = [2, 4, 6 ,8]
# num_quibts = [6 ,8]
sample_times = [0, 1, 2, 3]
mode = ["avg", "mode"]
seed = list(range(10))

for n in num_quibts:
    for st in sample_times:
        for m in mode:
            all_mse = []
            all_mae = []
            print("=====Args: num_qubits={}, sample_times={}, mode={}".format(n, st, m))
            for sd in seed:
                mse, mae = run_epoch(n, st, m, sd)
                all_mse.append(mse)
                all_mae.append(mae)
            print("\n++++MSE: mean: {:.4f}, std: {:.4f}".format(np.mean(all_mse), np.std(all_mse)))
            print("++++MAE: mean: {:.4f}, std: {:.4f}\n".format(np.mean(all_mae), np.std(all_mae)))