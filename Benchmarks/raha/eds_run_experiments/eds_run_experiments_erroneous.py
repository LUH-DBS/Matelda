import os
import logging

logging.basicConfig(filename='./data-gov-raha-results-erroneous.log', encoding='utf-8', level=logging.DEBUG)

repition = range(1, 11)
labeling_budgets = range(1, 21)
sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-sandbox"
results_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-raha-results-erroneous"

# sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/sandbox_test"
# results_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/sandbox_test-results-erroneous"

dir_levels = 1 # That means we have files in each subdirectory of sandbox dir
datasets = []

if dir_levels == 1:
    for dir in os.listdir(sandbox_path):
        datasets.append(os.path.join(sandbox_path , dir))

def run_raha(dataset, results_path, labeling_budget, exec_number):
    python_script = f'''python raha/detection.py \
                             --results_path {results_path} --base_path {dataset} --dataset {os.path.basename(dataset)} --labeling_budget {labeling_budget} --execution_number {exec_number}'''
    logging.warning(python_script)
    os.system(python_script)


for dataset in datasets:
    try:
        run_raha(dataset, results_path, 20, 1)
    except Exception as e:
        logging.error(dataset, e)