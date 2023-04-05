from multiprocessing import freeze_support
import os
import pickle

import pandas as pd
import ed_twolevel_rahas_features 
import saving_results
import app_logger


if __name__ == '__main__':

    cell_feature_generator_enabled = False
    noise_extraction_enabled = False
    sandbox_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/kaggle/separated_kaggle_lake"
    tables_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/kaggle/separated_kaggle_lake/kaggle_sample_sandbox"
    column_groups_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/mediate_files/col_grouping_res/"
    column_groups_df_path = os.path.join(column_groups_path, "col_df_res")
    column_groups_cpc_path = os.path.join(column_groups_path, "cols_per_clu")
    experiment_output_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/output"
    results_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/output/results"
    logs_dir = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/logs"
    n_table_groups = 4

    logger = app_logger.get_logger(logs_dir)
    labeling_budget = 100

    number_of_col_clusters = {}
    col_groups = 0
    total_col_groups = 0
    cluster_sizes = {}

    with open(os.path.join(results_path, 'tables_dict.pickle'), 'rb') as handle:
        tables_dict = pickle.load(handle, protocol=pickle.HIGHEST_PROTOCOL)

    for i in range(n_table_groups):
        path = os.path.join(column_groups_cpc_path, 'cols_per_cluster_{}.pkl'.format(i))
        path_labels = os.path.join(column_groups_df_path, 'col_df_labels_cluster_{}.pickle'.format(i))
        dict_ = pickle.load(open(path, 'rb'))
        dict_labels = pickle.load(open(path_labels, 'rb'))
        labels_df = pd.DataFrame.from_dict(dict_labels, orient='index').T
        col_clusters = set(labels_df['column_cluster_label'])
        number_of_col_clusters[str(i)] = len(col_clusters)
        cluster_sizes[str(i)] = {}
        for cc in col_clusters:
            cluster_sizes[str(i)][str(cc)] = 0
            df = labels_df[labels_df['column_cluster_label'] == cc]
            for idx, row in df.iterrows():
                cluster_sizes[str(i)][str(cc)] += len(row['col_value'])

    cell_clustering_alg= "km"

    print("starting error detection")
    
    y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
        unique_cells_local_index_collection, samples = \
            ed_twolevel_rahas_features.error_detector(cell_feature_generator_enabled, noise_extraction_enabled, sandbox_path, column_groups_df_path, experiment_output_path, results_path,\
                                                      labeling_budget, number_of_col_clusters, cluster_sizes, cell_clustering_alg, tables_dict)
    saving_results.get_all_results(tables_dict, tables_path, results_path, y_test_all, y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
    unique_cells_local_index_collection, samples)
