
import os 
import pickle
import pandas as pd

def get_res_df_eds(labeling_budgets, res_path, nested_dir_name, exp_name, exec):
    res_dict = {"labeling_budget": [], "precision": [], "recall": [], "fscore": []}
    for label_budget in labeling_budgets:
        print(label_budget)
        precision = 0
        recall = 0
        fscore = 0
        for exec_num in exec:
            path = f"{res_path}/{nested_dir_name}_{exec_num}/_{exp_name}_{label_budget}_labels/results"
            with open(os.path.join(path, "scores_all.pickle"), "rb") as f:
                scores_all = pickle.load(f)
                precision += scores_all["total_precision"] if scores_all["total_precision"] else 0
                recall += scores_all["total_recall"] if scores_all["total_recall"] else 0
                fscore += scores_all["total_fscore"] if scores_all["total_fscore"] else 0
        precision /= len(exec)
        recall /= len(exec)
        fscore /= len(exec)
        res_dict["labeling_budget"].append(label_budget)
        res_dict["precision"].append(precision)
        res_dict["recall"].append(recall)
        res_dict["fscore"].append(fscore)
    res_df_eds = pd.DataFrame(res_dict)
    return res_df_eds

executions = range(1, 6)
labeling_budget = [0.10, 0.25, 0.5, 0.75, 1, 2, 3]
res_path = "/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/Dgov_NCG_Exp"
exp_name = "NCG_output_lake_high_percent_processed"
nested_dir_name = "output_dgov"
n_cols = 768
for i in range (5, 21, 5):
    labeling_budget.append(i)
labeling_budgets_cells = [round(n_cols*x) for x in labeling_budget]
res_df_eds = get_res_df_eds(labeling_budgets_cells, res_path, nested_dir_name, exp_name, executions)
res_df_eds.to_csv(os.path.join("/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/final_csv_results", "DGov_EDS_NCG.csv"), index=False)