from collections import OrderedDict
import os
import pandas as pd
from operator import itemgetter
from multiprocessing import Pool

def run_raha(dataset, results_path, labeling_budget, exec_number):
    python_script = f'''python raha/detection.py \
                             --results_path {results_path} --base_path {dataset} --dataset {os.path.basename(dataset)} --labeling_budget {labeling_budget} --execution_number {exec_number}'''
    print(python_script)
    os.system(python_script)

def distribute_labels(labeling_budget_cells, dir_levels, sandbox_path):
    datasets = dict()
    datasets_budget = dict()
    datasets_num_cells = dict()
    datasets_actual_errors = dict()

    num_cols = 0
    num_rows = 0
    num_cells = 0

    if dir_levels == 1:
        for dir in os.listdir(sandbox_path):
            try:
                dataset_path = os.path.join(sandbox_path , dir)
                df = pd.read_csv(os.path.join(dataset_path, "dirty_clean.csv"), sep=",", header="infer", encoding="utf-8", dtype=str,
                                            keep_default_na=False, low_memory=False)
                actual_errors_df = pd.read_csv(os.path.join(dataset_path, "clean_changes.csv"), sep=",", header=None, encoding="utf-8", dtype=str,
                                            keep_default_na=False, low_memory=False)
                num_actual_errors = actual_errors_df.shape[0]
                datasets[dataset_path] = df.shape
                num_cells_dataset = df.shape[0] * df.shape[1]
                datasets_num_cells[dataset_path] = num_cells_dataset
                datasets_budget[dataset_path] = 0
                datasets_actual_errors[dataset_path] = num_actual_errors
                num_cols += df.shape[1]
                num_rows += df.shape[0]
                num_cells += num_cells_dataset
            except Exception as e:
                print(dir, e)    

    datasets_num_cells = OrderedDict(sorted(datasets_num_cells.items(), key=itemgetter(1), reverse=True))

    if labeling_budget_cells == num_cols:
        for dataset in datasets:
            datasets_budget[dataset] = 1 
    else:
        label_added = True
        asssigned_labels = 0 
        while labeling_budget_cells > asssigned_labels and label_added:
            label_added = False
            for dataset in datasets_num_cells.keys():
                dataset_num_cols = datasets[dataset][1]
                remained_labels = labeling_budget_cells - asssigned_labels
                if  remained_labels >= dataset_num_cols and datasets_actual_errors[dataset] > datasets_budget[dataset] + 1:
                    label_added = True
                    datasets_budget[dataset] += 1 
                    asssigned_labels += dataset_num_cols
    return datasets_budget 

def run_experiments(sandbox_path, results_path, dir_levels, labeling_budget_cells, exec_number):
    for label_budget in labeling_budget_cells:
        datasets_budget = distribute_labels(label_budget, dir_levels, sandbox_path)
        for dataset in datasets_budget:
            try:
                run_raha(dataset, results_path, datasets_budget[dataset], exec_number)
            except Exception as e:
                print(dataset, e)

def main():
    repition = range(1, 11)
    sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-sandbox"
    #sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/sandbox_test"
    results_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-raha-results-2"
    dir_levels = 1 # That means we have files in each subdirectory of sandbox dir
    # num_all_cells = 10480897
    labeling_budget_cells = (100, 1000000, 100)
    args = [(sandbox_path, results_path, dir_levels, labeling_budget_cells, exec_number) for exec_number in repition]
    pool = Pool(processes=5)
    pool.starmap(run_experiments, args)


if __name__ == "__main__":
    main()

