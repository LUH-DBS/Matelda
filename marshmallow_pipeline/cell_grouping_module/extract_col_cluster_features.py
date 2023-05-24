
import logging
import os
import pickle
import pandas as pd
from sklearn.cluster import DBSCAN


logger = logging.getLogger()


def get_col_cluster_features(table_cluster: int, group_df: pd.DataFrame, features_dict: dict, table_group_dict: dict, gt_provided: bool = False):
    n_col_clusters = len(group_df['column_cluster_label'].unique())
    for cluster in range(n_col_clusters):
        X_temp = []
        y_temp = []
        X_temp_dict = dict()
        c_df = group_df[group_df['column_cluster_label'] == cluster]
        for index, row in c_df.iterrows():
            for cell_idx in range(len(row['col_value'])):
                X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                X_temp_dict[len(X_temp) - 1] = (row['table_id'], row['col_id'], cell_idx)
                if gt_provided:
                    y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')])
            
        table_group_dict["table_cluster"].append(table_cluster)
        table_group_dict["col_cluster"].append(cluster)
        table_group_dict["features_X"].append(X_temp)
        table_group_dict["features_X_dict"].append(X_temp_dict)
        if gt_provided:
            table_group_dict["y"].append(y_temp)
            table_group_dict["column_group_error_ratio"].append(y_temp.count(1) / len(y_temp))
        else:
            table_group_dict["y"].append(None)
            table_group_dict["column_group_error_ratio"].append(None)
    return table_group_dict
