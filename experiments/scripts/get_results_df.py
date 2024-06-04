
import os 
import pickle
import pandas as pd

def get_res_df_eds(labeling_budgets, res_path, nested_dir_name, exp_name, exec):
    res_dict = {"labeling_budget": [], "used_labels": [], "precision": [], "recall": [], "fscore": [], "time": [], "tp": [], "fp": [], "fn": [], "tn": []}
    for label_budget in labeling_budgets:
        print(label_budget)
        precision = 0
        recall = 0
        fscore = 0
        time = 0
        used_labels = []
        for exec_num in exec:
            path = f"{res_path}/{nested_dir_name}_{exec_num}/_{exp_name}_{label_budget}_labels/results"
            with open(os.path.join(path, "time.txt"), "rb") as f:
                t = float(f.read())
            with open(os.path.join(path, "labeled_by_user.pickle"), "rb") as f:
                labeled_by_user = pickle.load(f)
                n_labels = 0
                for _, labels in labeled_by_user.items():
                    n_labels += len(labels) if labels else 0
                used_labels.append(n_labels)
            with open(os.path.join(path, "scores_all.pickle"), "rb") as f:
                scores_all = pickle.load(f)
                precision += scores_all["total_precision"] if scores_all["total_precision"] else 0
                recall += scores_all["total_recall"] if scores_all["total_recall"] else 0
                fscore += scores_all["total_fscore"] if scores_all["total_fscore"] else 0
                time += t
            total_tp = scores_all["total_tp"] if scores_all["total_tp"] else 0
            total_fp = scores_all["total_fp"] if scores_all["total_fp"] else 0
            total_fn = scores_all["total_fn"] if scores_all["total_fn"] else 0
            total_tn = scores_all["total_tn"] if scores_all["total_tn"] else 0

        precision /= len(exec)
        recall /= len(exec)
        fscore /= len(exec)
        time /= len(exec)
        res_dict["labeling_budget"].append(label_budget)
        res_dict["precision"].append(precision)
        res_dict["recall"].append(recall)
        res_dict["fscore"].append(fscore)
        res_dict["time"].append(time)
        res_dict["used_labels"].append(used_labels)
        res_dict["tp"].append(total_tp)
        res_dict["fp"].append(total_fp)
        res_dict["fn"].append(total_fn)
        res_dict["tn"].append(total_tn)
    res_df_eds = pd.DataFrame(res_dict)
    return res_df_eds

executions = range(0, 2)
# labeling_budget = [0.10, 0.25, 0.5, 0.75, 1, 2, 3]
labeling_budget = [0.10, 0.25, 0.5, 0.75, 1, 2, 3]
res_path = "./output_DGov_NTR_edbt"
exp_name = "test_edbt_DGov_NTR"
nested_dir_name = "output_DGov_NTR"
n_cols = 1385
for i in range (5, 21, 5):
    labeling_budget.append(i)
labeling_budgets_cells = [round(n_cols*x) for x in labeling_budget]
res_df_eds = get_res_df_eds(labeling_budgets_cells, res_path, nested_dir_name, exp_name, executions)
res_df_eds.to_csv(os.path.join("/./output_DGov_NTR_edbt/", "test.csv"), index=False)