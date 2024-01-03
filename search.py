import argparse
import logging
import os
import pickle as p
import sys
import time
from copy import deepcopy
from nas_bench_graph.readbench import read
from nas_bench_graph.architecture import Arch
import matplotlib.pyplot as plt
import numpy as np

from utils import get_model_infos
from pruning_models.model_graph import *
from pruning_models.model_graph_flop import *
from utils import calculate_IGD_value, set_seed
from utils import get_front_0
from training_free_metrics import get_config_for_training_free_calculator, get_training_free_calculator

from Graph_untils.hpo import HP
from Graph_untils.data import *
id_arch = 0
OPS_LIST = {
    '0': 'gat',
    '1': 'gcn',
    '2': 'gin',
    '3': 'cheb',
    '4': 'sage',
    '5': 'arma',
    '6': 'graph',
    '7': 'fc',
    '8': 'skip'
}

def convert_keep_mask_2_arch_str(keep_mask):
    arch = []
    for i in range(len(keep_mask)):
        for j in range(len(keep_mask[i])):
            if keep_mask[i][j]:
                arch.append(j)
                break
    return arch

def evaluate_arch(ind, link, measure,  dataset0, dataset1, dataset2, epoch=None, use_log=False):
    gnn_list = [
    "gat",  # GAT with 2 heads
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  # k-GNN
    "fc",  # fully-connected
    "skip"  # skip connection
    ]

    #print(ind)
    arch = ind.copy()
    lk = list(link)
    operations_temp = list(arch[:])
    operations = [gnn_list[i] for i in operations_temp]
    
    arch_str = Arch(lk, operations)

    if operations_temp != [8, 8, 8, 8]:
            info0 = dataset0[arch_str.valid_hash()]
            info1 = dataset1[arch_str.valid_hash()]
            info2 = dataset2[arch_str.valid_hash()]

    if measure=='test-accuracy':
        if operations_temp == [8, 8, 8, 8]:
            return 0
        return (np.array(info0['dur'])[:, 2].max() + np.array(info1['dur'])[:, 2].max() + np.array(info2['dur'])[:, 2].max()) / 3
    elif measure=='para':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        return (info0['para'] + info1['para'] + info2['para']) / 3
        
    elif measure=='latency': 
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        return (info0['latency'] + info1['latency'] + info2['latency']) / 3
    elif measure=='valid-accuracy':
        if operations_temp == [8, 8, 8, 8]:
            return 0
        return (info0['dur'][epoch][1] + info1['dur'][epoch][1] + info2['dur'][epoch][1]) / 3
    elif measure=='train-loss':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = info0['dur'][epoch][3]
        loss1 = info1['dur'][epoch][3]
        loss2 = info2['dur'][epoch][3]
        return (loss0 + loss1 + loss2) / 3
    elif measure=='valid-loss':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = info0['dur'][epoch][4]
        loss1 = info1['dur'][epoch][4]
        loss2 = info2['dur'][epoch][4]
        return (loss0 + loss1 + loss2) / 3
    elif measure=='sotl':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = np.sum(np.array(info0['dur'])[:epoch, 3])
        loss1 = np.sum(np.array(info1['dur'])[:epoch, 3])
        loss2 = np.sum(np.array(info2['dur'])[:epoch, 3])
        return (loss0 + loss1 + loss2) / 3
    elif measure=='sovl':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = np.sum(np.array(info0['dur'])[:epoch, 4])
        loss1 = np.sum(np.array(info1['dur'])[:epoch, 4])
        loss2 = np.sum(np.array(info2['dur'])[:epoch, 4])
        return (loss0 + loss1 + loss2) / 3
    elif measure=='sotl-e':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = np.sum(np.array(info0['dur'])[epoch-e:epoch, 3])
        loss1 = np.sum(np.array(info1['dur'])[epoch-e:epoch, 3])
        loss2 = np.sum(np.array(info2['dur'])[epoch-e:epoch, 3])
        return (loss0 + loss1 + loss2) / 3
    elif measure=='sovl-e':
        if operations_temp == [8, 8, 8, 8]:
            return np.inf
        loss0 = np.sum(np.array(info0['dur'])[epoch-e:epoch, 4])
        loss1 = np.sum(np.array(info1['dur'])[epoch-e:epoch, 4])
        loss2 = np.sum(np.array(info2['dur'])[epoch-e:epoch, 4])
        return (loss0 + loss1 + loss2) / 3
    return None
    
def evaluate(final_opt_archs,final_link_archs, dataset0, dataset1, dataset2, pf, path_results,num_net):
    approximation_set = []
    approximation_front = []
    approximation_front_ = []
    total_pos_training_time = 0.0
    best_arch, best_test_acc = None, 0.0
    for i in range(len(final_opt_archs)):
        keep_mask = final_opt_archs[i]
        choose_link = list(final_link_archs[i])

        arch = convert_keep_mask_2_arch_str(keep_mask)
        best_val_acc = 0.0

        param = evaluate_arch(arch, choose_link, measure = 'para', epoch=200, use_log=False,dataset0 = dataset0, dataset1 = dataset1,dataset2 = dataset2)
        test_acc = evaluate_arch(arch, choose_link, measure = 'test-accuracy', epoch=200, use_log=False, dataset0 = dataset0, dataset1 = dataset1,dataset2 = dataset2)
        F = [1 - test_acc, param]
        if test_acc > best_test_acc:
            best_arch = arch
            best_link = choose_link
            best_test_acc = test_acc
        if test_acc != 0.0:
            arch = deepcopy([*choose_link , *arch])
            approximation_set.append([arch])
            approximation_front.append(F)

    approximation_front = np.round(np.array(approximation_front), 4)
    idx_front_0 = get_front_0(approximation_front)

    approximation_set = np.array(approximation_set)[idx_front_0]
    approximation_front = approximation_front[idx_front_0]
    approximation_front[:,0] = 1 - approximation_front[:,0]
    IGD = np.round(calculate_IGD_value(pareto_front=pf, non_dominated_front=approximation_front), 6)

    logging.info(f'Evaluate -> Done!\n')
    logging.info(f'IGD: {IGD}')
    logging.info(f'Best Architecture: {best_arch}')
    logging.info(f'Best_Link: {best_link}')
    logging.info(f'Best Architecture (performance): {np.round(best_test_acc, 2)}\n')
    rs = {
        'n_archs_evaluated': len(final_opt_archs),
        'approximation_set': approximation_set,
        'approximation_front': approximation_front,
        'best_arch_found': best_arch,
        'best_arch_found (performance)': np.round(best_test_acc, 2),
        'IGD': IGD,
    }
    p.dump(rs, open(f'{path_results}/results_evaluation_{num_net}.p', 'wb'))

    logging.info(f'--- Approximation set ---\n')
    for i in range(len(approximation_set)):
        logging.info(
            f'arch: {approximation_set[i]} - PARAM: {approximation_front[i][1]} - testing error: {approximation_front[i][0]}\n')

    return IGD, np.round(best_test_acc, 2)


def prune(tf_ind, path_data, path_results, seed, dataset, num_net, max_eval):

    list_arch_parent = [
        [[True, True, True, True, True, True, True, True, True],
         [True, True, True, True, True, True, True, True, True],
         [True, True, True, True, True, True, True, True, True],
         [True, True, True, True, True, True, True, True, True]]
    ]  # At beginning, activating all operations
    max_nPrunes = len(list_arch_parent[-1])
    i = 0
    hpdict = {"dropout": 0.0, "dim": 256, "num_cells": 1, "num_pre": 1, "num_pro": 1, "lr": 0.01, "wd": 5e-4, "optimizer": "Adam", "num_epochs": 500}
    hp = HP()
    for key in hpdict:
        setattr(hp, key, hpdict[key])
    link_list = [ 
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,1],
    [0,0,1,2],
    [0,0,1,3],
    [0,1,1,1],
    [0,1,1,2],
    [0,1,2,2],
    [0,1,2,3]
    ]
    
    global id_arch
    while i <= max_nPrunes - 1:
        logging.info(f'------- The {i + 1}-th pruning -------\n')
        list_arch_child = []
        F_arch_child = []
        for arch in list_arch_parent:
            arch_child = deepcopy(arch)
            arch_child[i] = [False, False, False, False, False, False, False, False, False]
            for j in range(len(arch_child[i])):
                link = link_list[np.random.randint(9)]
                config = get_config_for_training_free_calculator(search_space='NASBenchGraph', dataset=dataset,
                                                     seed=seed, path_data=path_data, num_train_per_class = 4,
                                                     link = link, hp = hp, in_dim = get_num_features(dataset), out_dim = get_num_classes(dataset), tf = tf_ind )
                tf_calculator = get_training_free_calculator(config=config, method_type=tf_ind)
                id_arch += 1
                if id_arch > max_eval - 1:
                    return None,None,True
                arch_child[i][j] = True
                list_arch_child.append(deepcopy(arch_child))

                flat_list = [item for sublist in list_arch_child[-1] for item in sublist]
                network = get_graph_flop_model_from_arch_str(keep_mask = flat_list, link = link, hp = hp, in_dim = get_num_features(dataset),out_dim = get_num_classes(dataset) , dname = dataset)

                flop, param = get_model_infos(config,network, dataset)

                tf_metric_value = tf_calculator.compute(keep_mask=flat_list)[tf_ind]
                F = [param, -tf_metric_value]
                logging.info(f'ID Arch: {id_arch}')
                logging.info(f'Keep mask:\n{list_arch_child[-1]}')
                logging.info(f'PARAM: {F[0]}')
                logging.info(f'Synflow: {F[-1]}\n')
                F_arch_child.append(F)
                arch_child[i][j] = False
        idx_front_0 = get_front_0(F_arch_child)
        list_arch_parent = np.array(deepcopy(list_arch_child))[idx_front_0]
        logging.info(f'Number of architectures for the next pruning: {len(list_arch_parent)}\n')
        i += 1

    F_arch = []
    list_arch_chose = []
    list_link_chose = []
    for arch in list_arch_parent:
        flat_list = [item for sublist in arch for item in sublist]
        for link in link_list:
            id_arch += 1
            config = get_config_for_training_free_calculator(search_space='NASBenchGraph', dataset=dataset,
                                                        seed=seed, path_data=path_data, num_train_per_class = 4,
                                                        link = link, hp = hp, in_dim = get_num_features(dataset), out_dim = get_num_classes(dataset), tf = tf_ind )
            tf_calculator = get_training_free_calculator(config=config, method_type=tf_ind)

            network = get_graph_flop_model_from_arch_str(keep_mask = flat_list, link = link, hp = hp, in_dim = get_num_features(dataset), out_dim = get_num_classes(dataset) , dname = dataset)

            flop, param = get_model_infos(config,network, dataset)

            tf_metric_value = tf_calculator.compute(keep_mask=flat_list)[tf_ind]
            F = [param, -tf_metric_value]
            F_arch.append(F)
            list_arch_chose.append(deepcopy(arch))
            list_link_chose.append(deepcopy(link))

    idx_front_0 = get_front_0(F_arch)
    final_opt_archs = np.array(deepcopy(list_arch_chose))[idx_front_0]
    final_link_archs = np.array(deepcopy(list_link_chose))[idx_front_0]
            
    rs = {
        'final_opt_archs': final_opt_archs,
        'final_link_archs': final_link_archs,
        'num_eval' : id_arch
    }

    p.dump(rs, open(f'{path_results}/pruning_results_{num_net}.p', 'wb'))
    return final_opt_archs,final_link_archs,False


def main(kwargs):
    n_runs = kwargs.n_runs
    init_seed = kwargs.seed
    start_run = kwargs.start_run
    dataset_name = kwargs.dataset_name
    max_eval = kwargs.max_eval
    random_seeds_list = [init_seed + run * 100 for run in range(n_runs)]

    if kwargs.path_data is None:
        path_data = './benchmark_data'
    else:
        path_data = kwargs.path_data
    if kwargs.path_pareto is None:
        path_pareto = './pareto'
    else:
        path_pareto = kwargs.path_data
    
    if kwargs.path_results is None:
        path_results = './results/{}'.format(dataset_name)
    else:
        path_results = kwargs.path_results
    tf_metric = 'synflow'  

    bench0 = read(f'{path_data}/{dataset_name}0.bench')
    bench1 = read(f'{path_data}/{dataset_name}1.bench')
    bench2 = read(f'{path_data}/{dataset_name}2.bench')
    pareto_opt_front = p.load(open(f'{path_pareto}/{dataset_name}_pareto.pickle', "rb"))

    logging.info(f'******* PROBLEM *******')
    logging.info(f'- Benchmark: NAS-Bench-Graph')
    logging.info(f'- Dataset: {dataset_name}\n')

    logging.info(f'******* RUNNING *******')
    logging.info(f'- Pruning:')
    logging.info(f'\t+ The first objective (minimize): PARAM')
    logging.info(f'\t+ The second objective (minimize): -Synflow')

    logging.info(f'- Evaluate:')
    logging.info(f'\t+ The first objective (minimize): PARAM')
    logging.info(f'\t+ The second objective (minimize): test error\n')

    logging.info(f'******* ENVIRONMENT *******')
    logging.info(f'- Path for saving results: {path_results}\n')

    final_IGD_lst = []
    best_acc_found_lst = []

    for run_i in range(start_run,n_runs):
        logging.info(f'Run ID: {run_i + 1}')
        sub_path_results = path_results + '/' + f'{run_i}'

        global id_arch
        id_arch = 0 

        try:
            os.mkdir(sub_path_results)
        except FileNotFoundError:
            pass
        logging.info(f'Path for saving results: {sub_path_results}')

        random_seed = random_seeds_list[run_i]
        logging.info(f'Random seed: {run_i}')
        set_seed(random_seed)
        num_net = 0
        s = time.time()
        while id_arch < max_eval-1:
            opt_archs,link_archs,check = prune(tf_ind=tf_metric, path_data=path_data, path_results=sub_path_results, seed=random_seed,dataset = dataset_name,num_net = num_net,max_eval = max_eval)
            executed_time = time.time() - s
            if check == True:
                break
            logging.info(f'Prune Done! Execute in {executed_time} seconds.\n')
            p.dump(executed_time, open(f'{sub_path_results}/running_time_{num_net}.p', 'wb'))
            if num_net == 0:
                final_opt_archs = opt_archs
                final_link_archs = link_archs
            else:
                final_opt_archs = np.vstack((final_opt_archs,opt_archs))
                final_link_archs = np.vstack((final_link_archs,link_archs))
            IGD, best_acc = evaluate(final_opt_archs=final_opt_archs,final_link_archs=final_link_archs, dataset0 = bench0, dataset1 = bench1, dataset2 = bench2, pf=pareto_opt_front,
                                    path_results=sub_path_results,num_net = num_net)
            final_IGD_lst.append(IGD)
            best_acc_found_lst.append(best_acc)
            num_net += 1

    logging.info(f'Average IGD: {np.round(np.mean(final_IGD_lst), 4)} ({np.round(np.std(final_IGD_lst), 4)})')
    logging.info(
        f'Average best test-accuracy: {np.round(np.mean(best_acc_found_lst), 4)} ({np.round(np.std(best_acc_found_lst), 4)})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Grouped Evaluation Strategy for Training-Free Multi-Objective Pruning-based Graph Neural Architecture Search")

    ''' ENVIRONMENT '''
    parser.add_argument('--path_data', type=str, default=None, help='path for loading benchmark data')
    parser.add_argument('--path_results', type=str, default=None, help='path for saving results')
    parser.add_argument('--path_pareto', type=str, default=None, help='path for pareto front')
    parser.add_argument('--n_runs', type=int, default=30, help='number of experiment runs')
    parser.add_argument('--start_run', type=int, default=0, help='continue from the last run')
    parser.add_argument('--max_eval', type=int, default=1000, help='number of evaluations max')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--dataset_name', type=str, default='Computers', help='Name of dataset [Computers, Photo, CS, Cora, PubMed, CiteSeer]')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    main(args)
