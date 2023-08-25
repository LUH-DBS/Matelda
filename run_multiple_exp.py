import os
import shutil
import pipeline

executions = range(1, 6)
labeling_budget = [0.10, 0.25, 0.5, 0.75, 1, 2, 3]
n_cols = 768
for i in range (5, 21, 5):
    labeling_budget.append(i)
labeling_budgets_cells = [round(n_cols*x) for x in labeling_budget]

for execution in executions:
    for labeling_budget in labeling_budgets_cells:
        # directories_to_remove = [
        #     "marshmallow_pipeline/santos/benchmark/*",
        #     "marshmallow_pipeline/santos/stats/*",
        #     "marshmallow_pipeline/santos/hashmap/*",
        #     "marshmallow_pipeline/santos/groundtruth/*",
        #     "results"
        # ]

        # for directory in directories_to_remove:
        #     if os.path.exists(directory):
        #         shutil.rmtree(directory)
        print(labeling_budget, execution)
        try:
            pipeline.main(labeling_budget, execution)
        except:
            print("Error")
            print(labeling_budget, execution)
            continue

        

