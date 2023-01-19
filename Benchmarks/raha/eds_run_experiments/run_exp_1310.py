import argparse
import os
import logging

def run_raha(dataset, results_path, labeling_budget, exec_number):
    python_script = f'''python Benchmarks/raha/detection.py \
                             --results_path {results_path} --base_path {dataset} --dataset {os.path.basename(dataset)} --labeling_budget {labeling_budget} --execution_number {exec_number}'''
    logging.warning(python_script)
    os.system(python_script)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--labeling_budget", help="labeling_budget")
    parser.add_argument("--execution_number", help="execution_number")
    args = parser.parse_args()
    #labeling_budget = int(args.labeling_budget)
    execution_number = int(args.execution_number)


    sandbox_path = "/home/fatemeh/ED-Scale/Sandbox_Generation/data-gov-sandbox"
    results_path = "/home/fatemeh/ED-Scale/results/raha"

    for labeling_budget in range(1,4):
        res_dir_path = os.path.join(results_path, f'''{labeling_budget}_labels_{execution_number}_execution''')
        log_path = os.path.join("/home/fatemeh/ED-Scale/results/raha/logs", f'''{labeling_budget}_labels_{execution_number}_execution.log''')

        logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)

        dir_levels = 1 # That means we have files in each subdirectory of sandbox dir
        datasets = []

        if dir_levels == 1:
            for dir in os.listdir(sandbox_path):
                datasets.append(os.path.join(sandbox_path , dir))

        if not os.path.exists(res_dir_path):
            os.makedirs(res_dir_path)
            for dataset in datasets:
                try:
                    run_raha(dataset, res_dir_path, labeling_budget, execution_number)
                except Exception as e:
                    logging.error(dataset, e)
        else:
            logging.warn("directory exists")

if __name__ == "__main__":
    main()