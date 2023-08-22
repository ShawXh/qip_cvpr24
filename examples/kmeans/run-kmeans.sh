# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 1 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 2 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 3 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 1 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 2 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 3 --out-strategy avg

# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 1 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 2 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 3 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 1 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 2 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 3 --out-strategy avg

# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 1 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 2 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 3 --out-strategy mode
# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 1 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 2 --out-strategy avg
# python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 3 --out-strategy avg



function run() {
    python run-kmeans.py --mode classical --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 3 --out-strategy mode --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 4 --sample-times 3 --out-strategy avg --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 3 --out-strategy mode --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 6 --sample-times 3 --out-strategy avg --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 3 --out-strategy mode --seed $seed
    # python run-kmeans.py --mode quantum --num-qubits 8 --sample-times 3 --out-strategy avg --seed $seed
    python run-kmeans.py --mode quantum --num-qubits 10 --sample-times 3 --out-strategy mode --seed $seed
    python run-kmeans.py --mode quantum --num-qubits 10 --sample-times 3 --out-strategy avg --seed $seed
}

seed=0
run
seed=1
run
seed=2
run
seed=3
run
seed=4
run

