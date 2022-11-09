import os

repition = range(1, 11)
datasets = ["beers", "flights", "hospital", "movies_1", "rayyan", "tax"]
labeling_budgets = range(1, 21)

for exec_number in repition:
    for dataset in datasets:
        for label_budget in labeling_budgets:
            python_script = "python Benchmarks/raha/detection.py \
                             --dataset {} --labeling_budget {} --execution_number {}".format(dataset, label_budget, exec_number)
            print(python_script)
            os.system(python_script)