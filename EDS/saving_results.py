import logging
import pickle
import os
import pandas as pd
import app_logger

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger()

def get_classification_results(y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples):
    logger.info("Classification results:")
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    for i in predicted_all.keys():
        col_cluster_prediction = list(predicted_all[i])
        col_cluster_y = y_test_all[i]

        # TODO: Fix this, user labels are not always accurate
        col_cluster_prediction.extend(y_labeled_by_user_all[i])
        col_cluster_y.extend(y_labeled_by_user_all[i])

        tn, fp, fn, tp = confusion_matrix(y_true=col_cluster_y, y_pred=col_cluster_prediction, labels=[0,1]).ravel()

        total_tn += tn
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision, recall, f_score, support = precision_recall_fscore_support(col_cluster_y, col_cluster_prediction, average='macro')
        logger.info("col_cluster: {}, tn: {}, fp: {}, fn: {}, tp: {}".format(i, tn, fp, fn, tp))
        scores = {"col_cluster": i, "precision": precision, "recall": recall, "f_score": f_score, "support": support, "tp": tp, "fp": fp,
                "fn": fn, "tn": tn}
        with open(os.path.join(results_dir, "scores_col_cluster_{}.pickle".format(i)), "wb") as file:
            pickle.dump(scores, file)

    total_precision = total_tp/(total_tp + total_fp)
    total_recall = total_tp/(total_tp + total_fn)
    total_fscore = (2 * total_precision * total_recall) / (total_precision + total_recall)
    total_scores = {"n_samples": len(samples), 
                    "total_recall": total_recall, 
                    "total_precision": total_precision, 
                    "total_fscore": total_fscore,
                    "total_tp": total_tp, "total_fp": total_fp, "total_tn": total_tn, "total_fn": total_fn}
    with open(os.path.join(results_dir, "scores_all.pickle"), "wb") as file:
            pickle.dump(total_scores, file)
    return 

def create_predictions_dict(all_tables_dict, y_test_all, \
                    y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                    unique_cells_local_index_collection, samples):
    logger.info("get_predictions_dict")
    rows_list = []
    for key in unique_cells_local_index_collection.keys():
        logger.info("key: {}".format(key))
        for cell_key in list(unique_cells_local_index_collection[key].keys()):
            try:
                cell_local_idx = unique_cells_local_index_collection[key][cell_key]
                y_cell_ids = y_local_cell_ids[key]
                y_local_idx = y_cell_ids.index(cell_local_idx) if cell_local_idx in y_cell_ids else -1
                tmp_dict = {'table_id': cell_key[0], 'table_name': all_tables_dict[cell_key[0]]['name'], 
                'table_shape': all_tables_dict[cell_key[0]]['shape'],
                        'col_id': cell_key[1], 'col_name': all_tables_dict[cell_key[0]]['schema'][cell_key[1]], 
                        'cell_idx': cell_key[2], 'cell_value': cell_key[3],
                        'predicted': predicted_all[key][y_local_idx] if y_local_idx != -1 else -1,
                        'label': y_test_all[key][y_local_idx] if y_local_idx != -1 else samples[(key[0], key[1], cell_local_idx)]}
                rows_list.append(tmp_dict)
            except Exception as e:
                logger.info(e)
    results_df = pd.DataFrame(rows_list,
                              columns=['table_id', 'table_name', 'table_shape', 'col_id', 'col_name', 'cell_idx', 'cell_value',
                                       'predicted', 'label'])                 
    return results_df


def get_results_per_table(result_df):
    logger.info("get_results_per_table")
    results_per_table = dict()
    for table_id in result_df['table_id'].unique():
        table_df = result_df[result_df['table_id'] == table_id]
        table_name = table_df['table_name'].unique()[0]
        table_shape = table_df['table_shape'].unique()[0]
        tp = len(table_df[(table_df['predicted'] == 1) & (table_df['label'] == 1)])
        fp = len(table_df[(table_df['predicted'] == 1) & (table_df['label'] == 0)])
        fn = len(table_df[(table_df['predicted'] == 0) & (table_df['label'] == 1)])
        tn = len(table_df[(table_df['predicted'] == 0) & (table_df['label'] == 0)])
        precision = tp/(tp + fp) if tp + fp != 0 else 0
        recall = tp/(tp + fn) if tp + fn != 0 else 0
        f_score = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
        results_per_table[table_id] = {"table_name": table_name, "table_shape": table_shape, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                                       "precision": precision, "recall": recall, "f_score": f_score}
    return results_per_table

def get_tables_dict(sandbox_path):
    logger.info("get_tables_dict")
    all_tables_dict = {}
    table_id = 0
    table_dirs = os.listdir(sandbox_path)
    table_dirs.sort()
    for table in table_dirs:
        if not table.startswith("."):
            table_path = os.path.join(sandbox_path, table)
            table_df = pd.read_csv(table_path + "/dirty_clean.csv", sep=",", header="infer", encoding="utf-8",
                                        dtype=str, low_memory=False)
            table_df = table_df.applymap(lambda x: x.replace('"', '') if isinstance(x, str) else x)
            all_tables_dict[table_id] = {"name": table, "schema": table_df.columns.tolist(), "shape": table_df.shape}
            table_id += 1
    return all_tables_dict


def get_all_results(tables_path, results_dir, y_test_all, \
                    y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                    unique_cells_local_index_collection, samples):
    with open(os.path.join(results_dir, "labeled_by_user.pickle"), "wb") as file:
            pickle.dump(y_labeled_by_user_all, file)
    logger.info("getting classification results")
    tables_dict = get_tables_dict(tables_path)
    get_classification_results(y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples)
    results_df = create_predictions_dict(tables_dict, y_test_all, \
                    y_local_cell_ids, predicted_all, y_labeled_by_user_all,\
                    unique_cells_local_index_collection, samples)
    results_per_table = get_results_per_table(results_df)
    logger.info("Saving results_df")
    with open(os.path.join(results_dir, "results_df.pickle"), "wb") as file:
            pickle.dump(results_df, file)
    with open(os.path.join(results_dir, "results_per_table.pickle"), "wb") as file:
            pickle.dump(results_per_table, file)   
    logger.info("All done :)")
    return 
    
