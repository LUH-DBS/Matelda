import os
import pickle

import pandas as pd

from get_lake_info import get_info

def test_n_cells_evaluated(scores_all, actual_n_cells):
    n_cells = 0
    n_cells += scores_all["total_tp"]
    n_cells += scores_all["total_fp"]
    n_cells += scores_all["total_fn"]
    n_cells += scores_all["total_tn"]
    try:
        assert n_cells == actual_n_cells
        print("Test test_n_cells_evaluated passed!")
    except AssertionError:
        print("AssertionError test_n_cells_evaluated: n_cells = {} != {}".format(n_cells, actual_n_cells))
        raise AssertionError
    
def test_n_errors_considered(scores_all, actual_n_errors):
    n_errors = 0
    n_errors += scores_all["total_tp"]
    n_errors += scores_all["total_fn"]
    try:
        assert n_errors == actual_n_errors
        print("Test test_n_errors_considered passed!")
    except AssertionError:
        print("AssertionError test_n_errors_considered: n_errors = {} != {}".format(n_errors, actual_n_errors))
        raise AssertionError

def test_measures_calculated(scores_all):
    try:
        precision = scores_all["total_tp"] / (scores_all["total_tp"] + scores_all["total_fp"]) if scores_all["total_tp"] + scores_all["total_fp"] > 0 else None
        recall = scores_all["total_tp"] / (scores_all["total_tp"] + scores_all["total_fn"]) if scores_all["total_tp"] + scores_all["total_fn"] > 0 else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None else None
        assert scores_all["total_precision"] == precision
        assert scores_all["total_recall"] == recall
        assert scores_all["total_fscore"] == f1
        print("Test test_measures_calculated passed!")
    except AssertionError:
        print("AssertionError test_measures_calculated: precision = {}, recall = {}, f1 = {} != {}, {}, {}".format(
            scores_all["total_precision"], scores_all["total_recall"], scores_all["total_fscore"], precision, recall, f1))
        raise AssertionError

def test_col_groups_res(res_base_path, scores_all):
    res = []
    list_files = os.listdir(res_base_path)
    for l in list_files:
        if l.startswith("scores_col_cluster"):
            with open(os.path.join(res_base_path, l), 'rb') as f:
                scores = pickle.load(f)
                res.append(scores)
    scores_df = pd.DataFrame(res)
    try:
        assert scores_df["tp"].sum() == scores_all["total_tp"]
        assert scores_df["fp"].sum() == scores_all["total_fp"]
        assert scores_df["fn"].sum() == scores_all["total_fn"]
        assert scores_df["tn"].sum() == scores_all["total_tn"]
        print("Test test_col_groups_res passed!")
    except AssertionError:
        print("AssertionError test_col_groups_res: tp = {}, fp = {}, fn = {}, tn = {} != {}, {}, {}, {}".format(
            scores_df["tp"].sum(), scores_df["fp"].sum(), scores_df["fn"].sum(), scores_df["tn"].sum(),
            scores_all["total_tp"], scores_all["total_fp"], scores_all["total_fn"], scores_all["total_tn"]))
        raise AssertionError


def test_col_group_test_cells(res_base_path, mediate_file_path):
    col_df_path = os.path.join(mediate_file_path, "col_grouping_res", "col_df_res")
    list_col_dfs = os.listdir(col_df_path)
    for l in list_col_dfs:
        if l.startswith("col_df_labels_cluster"):
            table_cluster = l.removeprefix("col_df_labels_cluster_").removesuffix(".pickle").split("_")[0]
            with open(os.path.join(col_df_path,l), 'rb') as f:
                col_dict = pickle.load(f)
                col_df = pd.DataFrame(col_dict)
                for col_cluster in col_df["column_cluster_label"].unique():
                    col_cluster_df = col_df[col_df["column_cluster_label"] == col_cluster]
                    n_cells = 0
                    for i, row in col_cluster_df.iterrows():
                        n_cells += len(row["col_value"])
                    with open(os.path.join(res_base_path, "scores_col_cluster_{}_{}.pickle".format(table_cluster, col_cluster)), 'rb') as f:
                        scores = pickle.load(f)
                        try:
                            assert n_cells == scores["tp"] + scores["fp"] + scores["fn"] + scores["tn"]
                            print("table_cluster: {}, col_cluster: {}".format(table_cluster, col_cluster))
                            print("Test test_col_group_test_cells passed!")
                        except AssertionError:
                            print("table_cluster: {}, col_cluster: {}".format(table_cluster, col_cluster))
                            print("AssertionError test_col_group_test_cells: n_cells = {} != {}".format(n_cells, scores["tp"] + scores["fp"] + scores["fn"] + scores["tn"]))
                            raise AssertionError

sandbox_base_path = "datasets"                             
sandbox_name = "Quintet"           
dirty_file_names = "dirty.csv"
clean_file_names = "clean.csv"
scores_all_path = "output_quintet_1/_spell_checker_Quintet_2000_labels/results/scores_all.pickle"
res_base_path = "output_quintet_1/_spell_checker_Quintet_2000_labels/results"
mediate_file_path = "output_quintet_1/_spell_checker_Quintet_2000_labels/mediate_files"

with open(scores_all_path, 'rb') as f:
        scores_all = pickle.load(f)
test_n_cells_evaluated(scores_all, get_info(sandbox_base_path, sandbox_name, dirty_file_names, clean_file_names)[4])
test_n_errors_considered(scores_all, get_info(sandbox_base_path, sandbox_name, dirty_file_names, clean_file_names)[5])
test_measures_calculated(scores_all)
test_col_groups_res(res_base_path, scores_all)
test_col_group_test_cells(res_base_path, mediate_file_path)
    
