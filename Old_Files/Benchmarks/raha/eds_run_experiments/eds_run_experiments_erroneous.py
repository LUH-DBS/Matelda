import os
import logging

logging.basicConfig(filename='./data-gov-raha-results-kaggle.log', encoding='utf-8', level=logging.DEBUG)

repition = range(1, 11)
labeling_budgets = range(1, 21)
sandbox_path = "/home/fatemeh/ED-Scale/Old_Files/Benchmarks/kaggle_sample_dataset/separated_kaggle_lake/kaggle_sample_sandbox"
results_path = "/home/fatemeh/ED-Scale/Old_Files/Benchmarks/kaggle_sample_dataset/results"

dir_levels = 1 # That means we have files in each subdirectory of sandbox dir
datasets = []

if dir_levels == 1:
    for dir in os.listdir(sandbox_path):
        datasets.append(os.path.join(sandbox_path , dir))

def run_raha(dataset, results_path, labeling_budget, exec_number):
    python_script = f'''python /home/fatemeh/ED-Scale/Old_Files/Benchmarks/raha/detection.py \
                             --results_path {results_path} --base_path {dataset} --dataset {os.path.basename(dataset)} --labeling_budget {labeling_budget} --execution_number {exec_number}'''
    print(python_script)
    os.system(python_script)


for dataset in datasets:
    try:
        run_raha(dataset, results_path, 2, 2)
    except Exception as e:
        logging.error(dataset, e)