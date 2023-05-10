
import logging
import os
import pickle
import pandas as pd
from sklearn.cluster import DBSCAN


logger = logging.getLogger()

def get_col_cluster_noise(table_cluster, group_df, features_dict, noise_dict):
    logger.info("get_col_cluster_noise")
    n_col_clusters = len(group_df['column_cluster_label'].unique())
    for cluster in range(n_col_clusters):
        X_temp = []
        y_temp = []
        X_temp_dict = dict()
        noise_status = dict()
        c_df = group_df[group_df['column_cluster_label'] == cluster]
        for index, row in c_df.iterrows():
            for cell_idx in range(len(row['col_value'])):
                X_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                y_temp.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')])
                X_temp_dict[len(X_temp) - 1] = (row['table_id'], row['col_id'], cell_idx)
            
        clustering = DBSCAN(min_samples = 2).fit(X_temp)
        labels = clustering.labels_
        cells_per_db_clusters = dict()
        for i in range(len(labels)):
            if labels[i] not in cells_per_db_clusters:
                cells_per_db_clusters[labels[i]] = []
            cells_per_db_clusters[labels[i]].append(X_temp_dict[i])
        error_ratio = dict()
        for key in cells_per_db_clusters.keys():
            if key == -1:
                continue
            error_ratio[key] = 0
            for cell in cells_per_db_clusters[key]:
                if features_dict[(cell[0], cell[1], cell[2], 'gt')] == 1:
                    error_ratio[key] += 1
            if y_temp.count(1) > 0:
                error_ratio[key] /= y_temp.count(1)
            else:
                error_ratio[key] = None
        noise_count = 0
        for l in labels:
            if l == -1:
                noise_count += 1
        for i in range(len(labels)):
            if labels[i] == -1:
                noise_status[X_temp_dict[i]] = True
            else:
                noise_status[X_temp_dict[i]] = False
        noise_dict["table_cluster"].append(table_cluster)
        noise_dict["col_cluster"].append(cluster)
        noise_dict["n_noise"].append(noise_count)
        noise_dict["noise_status"].append(noise_status)
        noise_dict["error_ratio"].append(error_ratio)
        noise_dict["modularity"].append(c_df["modularity"].iloc[0])
        noise_dict["column_group_error_ratio"].append(y_temp.count(1) / len(y_temp))
        noise_dict["community_avg_degree_avg"].append(c_df["community_avg_degree"].iloc[0])
        noise_dict["community_size_avg"].append(c_df["community_size"].iloc[0])
        logger.info("DBSCAN table cluster {} col cluster {} done - n_noise: {}".format(table_cluster, cluster, noise_count))
            
    return noise_dict

def extract_noise(col_groups_dir, output_path, features_dict):
    noise_dict = {"table_cluster": [], "col_cluster": [], "n_noise": [], "noise_status":[], "error_ratio":[], "modularity":[], "column_group_error_ratio":[], "community_avg_degree_avg":[], "community_size_avg":[]}
    for file_name in os.listdir(col_groups_dir):
        if ".pickle" in file_name:
            file = open(os.path.join(col_groups_dir, file_name), 'rb')
            group_df = pickle.load(file)
            if not isinstance(group_df, pd.DataFrame):
                group_df = pd.DataFrame.from_dict(group_df, orient='index').T
            table_cluster = file_name.removeprefix("col_df_labels_cluster_").removesuffix(".pickle")
            file.close()
            noise_dict = get_col_cluster_noise(table_cluster, group_df, features_dict, noise_dict)
    with open(os.path.join(output_path, "noise_dict.pickle"), "wb") as filehandler:
        pickle.dump(noise_dict, filehandler)
    logger.info("Noise extraction finished.")
    return noise_dict