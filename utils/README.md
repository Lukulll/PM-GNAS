
# Grouped Evaluation Strategy for Training-Free Multi-Objective Pruning-based Graph Neural Architecture Search

Luc Truong, An Vo, Khoa Huu Tran, and Ngoc Hoang Luong

## Setup

- Clone this repository
- Install packages
```
$ pip install -r requirements.txt
```
- Download [NAS-Bench-Graph](https://figshare.com/articles/dataset/NAS-bench-Graph/20070371), put it in the `benchmark_data` folder and follow instructions [here](https://github.com/THUMNLab/NAS-Bench-Graph)

## Usage
To run the code, use the command below with the required arguments

```shell
python search.py --dataset_name <dataset_name> --n_runs <number_of_runs> --max_eval <number_of_evaluations> 
```

Example commands:
```shell
# GES-TF-MOPGNAS for Computers dataset which 30 run and stop when pass 1000 of evaluations
python search.py --dataset Computers --n_runs 30 ----max_eval 1000 
```

## Acknowledgement
Our source code is inspired by:

- [NATS-Bench: Benchmarking NAS Algorithms for Architecture Topology and Size](https://github.com/D-X-Y/NATS-Bench)
- [Zero-Cost Proxies for Lightweight NAS](https://github.com/SamsungLabs/zero-cost-nas)
- [TF-MOPNAS: Training-free multi-objective pruning-based neural architecture search](https://github.com/ELO-Lab/TF-MOPNAS)
