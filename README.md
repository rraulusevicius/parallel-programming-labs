# Parallel Programming Labs

Coursework for the **Lygiagretusis programavimas** (Parallel Programming) course at Kaunas University of Technology (KTU), 3rd year — shared-memory threads, OpenMP, message passing with Go channels, CUDA, and a Python multiprocessing project.

All labs process the same kind of data: read a JSON dataset, compute a value per item in parallel, filter by a condition and collect the results in sorted order.

## Lab 1 — Shared memory (C++)

Producer–consumer pattern with thread synchronization.

- **Lab1a — monitors:** data and result monitors guarded by a mutex, condition variables to block when the buffer is full/empty, worker threads that process and filter items, results kept sorted
- **Lab1b — OpenMP:** the same problem with OpenMP critical sections, manual data distribution across threads and a parallel reduction for sums
- **Lab1g, Lab1Kontrolinis** — defence and in-class test exercises

**Concepts:** mutexes, condition variables, critical sections, producer–consumer

## Lab 2 — Distributed memory (Go)

Message passing with Go channels — no shared memory at all.

- Four process roles: main, data, result and workers, all communicating over unbuffered channels
- Workers request data, compute results and send them to the result process
- Same filtering and sorting logic as Lab 1
- **Lab 2g – 2g-6** — defence exercises with variations

**Concepts:** goroutines, channels, message passing

## Lab 4 — GPU (CUDA)

The data processing moved to the GPU: each thread computes a value for its slice of the items, results are appended to a shared output buffer using `atomicAdd` for unique write positions, then copied back and printed.

- `kernel.cu` — students dataset (`students.json`), the version built by the project
- `product_cuda.cu` — products dataset (`IF31_RauluseviciusR_L1_dat_*.json`) with a risk score per product; output in `IF31_RauluseviciusR_L4_rez.txt`

## Engineering project (IP) — Python multiprocessing

[`IP/`](IP/) parallelizes the neural-network price-prediction task from my Numerical Methods course (3 inputs → 4 neurons → 1 output, trained with steepest descent on house-sale data). Parallelizing the training of a *single* small model gave too much overhead, so the final version trains an **ensemble of models in parallel** with `multiprocessing.Pool` (one model per process) and keeps the best one.

| File | Purpose |
|---|---|
| `parallel_final.py` | Final version — benchmarks 1/2/4/6/8 processes on 8 dataset sizes (100 → 14 600 rows), 3 runs each, and plots time and speedup |
| `data_parallel_version.py` | Alternative tried: split the data across processes and combine gradients for one model |
| `ensemble_averaging.py` | Alternative tried: average the ensemble's predictions instead of picking the best model |
| `augment_data.py` | Generates the larger `data_augmented.tsv` from `data_for_task3.tsv` |
| `results/` | Charts, CSV results and performance reports for 4- and 8-model ensembles |

Best result: **3.66× speedup** with 8 processes on the largest dataset (16-core CPU).

![Speedup vs number of processes](IP/results/8%20modeliai%20res/02_speedup_vs_processes.png)

## Building and running

- **Lab 1 and Lab 4 (C++/CUDA):** open the `.sln` in Visual Studio 2022. The nlohmann/json dependency is restored from NuGet (`packages.config`) on the first build. Lab 4 needs the NVIDIA CUDA Toolkit and an NVIDIA GPU.
- **Lab 2 (Go):** `cd "Lab 2" && go run main.go` (each `Lab 2g-*` folder runs the same way).
- **IP (Python 3):**
  ```bash
  cd IP
  pip install -r requirements.txt
  python parallel_final.py
  ```

## Tech

C++20 with std::thread and OpenMP · Go (goroutines, channels) · CUDA · Python (multiprocessing, NumPy, pandas, matplotlib)
