
# PM-GNAS: A Pruning-Based Approach for Efficient Multi-Objective Graph Neural Architecture Search

Truong Luc, Khoa Huu, Ngoc Hoang Luong.




## Setup
- Clone this repo.
- Install necessary packages.
```
$ pip install -r requirements.txt
```
- Download the database in [here](https://figshare.com/articles/dataset/NAS-bench-Graph/20070371). Then put in ```benchmark_data``` folder.
***Note:*** If you search in another search space you need to modify the code in pruning_models to fit the search space. If you run on another benchmark, please create a new Pareto folder to calculate the indicators.
## Usage/Examples
To experiment in NAS-Bench-Graph on a specific dataset you can run the code by call the search.py.
```shell
python search.py --n_runs <the_number_of_experiment_runs, default: 31> --dataset_name <Name_of_dataset> --max_eval <number_of max_evaluations> 
```
For example, we want to search on Computers:
```shell
$ python search.py --dataset_name Computers
```


## Acknowledgements
the code is inspired by:
 - [TF-MOPNAS: Training-free multi-objective pruning-based neural architecture search](https://github.com/ELO-Lab/TF-MOPNAS#tf-mopnas-training-free-multi-objective-pruning-based-neural-architecture-search)
 - [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas)
 - [NAS-Bench-Graph](https://github.com/THUMNLab/NAS-Bench-Graph)

