import os
import pickle
import pandas as pd 


def loading_columns_grouping_results(table_grouping_dict, mediate_files_path: str):
    """
    Loading the results of the column grouping
    
    Args:
        table_grouping_dict (dict): dictionary of the table grouping results
        mediate_files_path (str): path to the mediate files directory
        
    Returns:
        number_of_col_clusters (dict): dictionary of the number of column clusters per table cluster
        cluster_sizes_dict (dict): dictionary of the number of cells per column cluster
        column_groups_df_path (str): path to the column grouping results directory
    """
    n_table_groups = len(table_grouping_dict)
    number_of_col_clusters = {}
    cluster_sizes_dict = {"table_cluster": [], "col_cluster": [], "n_cells": []}
    column_groups_path = os.path.join(mediate_files_path, "col_grouping_res")
    column_groups_df_path = os.path.join(column_groups_path, "col_df_res")

    for i in range(n_table_groups):
        path_labels = os.path.join(column_groups_df_path, f'col_df_labels_cluster_{i}.pickle')
        dict_labels = pickle.load(open(path_labels, 'rb'))
        labels_df = pd.DataFrame.from_dict(dict_labels, orient='index').T
        col_clusters = set(labels_df['column_cluster_label'])
        number_of_col_clusters[str(i)] = len(col_clusters)
        for cc in col_clusters:
            df = labels_df[labels_df['column_cluster_label'] == cc]
            n_cells = 0
            for _, row in df.iterrows():
                n_cells += len(row['col_value'])
            cluster_sizes_dict["table_cluster"].append(i)
            cluster_sizes_dict["col_cluster"].append(cc)
            cluster_sizes_dict["n_cells"].append(n_cells)

    return number_of_col_clusters, cluster_sizes_dict, column_groups_df_path
