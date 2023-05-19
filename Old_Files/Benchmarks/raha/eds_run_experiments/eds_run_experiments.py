import os

repition = range(1, 11)
labeling_budgets = range(1, 21)
sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-sandbox"
results_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-raha-results-1"
dir_levels = 1 # That means we have files in each subdirectory of sandbox dir
datasets = []

if dir_levels == 1:
    for dir in os.listdir(sandbox_path):
        datasets.append(os.path.join(sandbox_path , dir))

def run_raha(dataset, results_path, labeling_budget, exec_number):
    python_script = f'''python raha/detection.py \
                             --results_path {results_path} --base_path {dataset} --dataset {os.path.basename(dataset)} --labeling_budget {labeling_budget} --execution_number {exec_number}'''
    print(python_script)
    os.system(python_script)


for exec_number in repition:
    for dataset in datasets:
        for label_budget in labeling_budgets:
            try:
                run_raha(dataset, results_path, label_budget, exec_number)
            except Exception as e:
                print(dataset, e)