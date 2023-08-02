import os
import shutil
import pipeline


# labeling_budget_dgov = [0.25, 0.5, 0.75]
labeling_budget_dgov = []
for i in range (7, 21):
    labeling_budget_dgov.append(i)

labeling_budgets_dgov = [round(1511*x) for x in labeling_budget_dgov]

for labeling_budget in labeling_budgets_dgov:
    directories_to_remove = [
        "marshmallow_pipeline/santos/benchmark/*",
        "marshmallow_pipeline/santos/stats/*",
        "marshmallow_pipeline/santos/hashmap/*",
        "marshmallow_pipeline/santos/groundtruth/*",
        "results"
    ]

    for directory in directories_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    print(labeling_budget)
    try:
        pipeline.main(labeling_budget)
    except:
        print("Error")
        print(labeling_budget)
        continue

    

