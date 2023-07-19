import hashlib
import logging
import os
import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix

from marshmallow_pipeline.utils.read_data import read_csv


def get_classification_results(
    y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples
):
    logging.debug("Classification results:")
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    for i in predicted_all.keys():
        col_cluster_prediction = list(predicted_all[i])
        col_cluster_y = y_test_all[i]

        # TODO: Fix this, user labels are not always accurate
        col_cluster_prediction.extend(y_labeled_by_user_all[i])
        col_cluster_y.extend(y_labeled_by_user_all[i])

        tn, fp, fn, tp = confusion_matrix(
            y_true=col_cluster_y, y_pred=col_cluster_prediction, labels=[0, 1]
        ).ravel()

        total_tn += tn
        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        f_score = (
            (2 * precision * recall) / (precision + recall)
            if precision and recall
            else None
        )

        scores = {
            "col_cluster": i,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

        logging.info(scores)

        with open(
            os.path.join(results_dir, "scores_col_cluster_" + str(i[0]) + "_" + str(i[1]) + ".pickle"), "wb"
        ) as file:
            pickle.dump(scores, file)

    total_precision = total_tp / (total_tp + total_fp)
    total_recall = total_tp / (total_tp + total_fn)
    total_fscore = (2 * total_precision * total_recall) / (
        total_precision + total_recall
    )
    total_scores = {
        "n_samples": len(samples),
        "total_recall": total_recall,
        "total_precision": total_precision,
        "total_fscore": total_fscore,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_tn": total_tn,
        "total_fn": total_fn,
    }

    logging.info("Total scores: %s", total_scores)
    with open(os.path.join(results_dir, "scores_all.pickle"), "wb") as file:
        pickle.dump(total_scores, file)


def create_predictions_dict(
    all_tables_dict,
    y_test_all,
    y_local_cell_ids,
    predicted_all,
    unique_cells_local_index_collection,
    samples,
):
    logging.debug("Getting predictions dict")
    rows_list = []
    samp_count = 0
    for key in unique_cells_local_index_collection.keys():
        logging.debug("Key: %s", key)
        for cell_key in list(unique_cells_local_index_collection[key].keys()):
            try:
                cell_local_idx = unique_cells_local_index_collection[key][cell_key]
                y_cell_ids = y_local_cell_ids[key]
                y_local_idx = (
                    y_cell_ids.index(cell_local_idx)
                    if cell_local_idx in y_cell_ids
                    else -1
                )
                tmp_dict = {
                    "table_id": cell_key[0],
                    "table_name": all_tables_dict[cell_key[0]]["name"],
                    "table_shape": all_tables_dict[cell_key[0]]["shape"],
                    "col_id": cell_key[1],
                    "col_name": all_tables_dict[cell_key[0]]["schema"][cell_key[1]],
                    "cell_idx": cell_key[2],
                    "cell_value": cell_key[3],
                    "predicted": predicted_all[key][y_local_idx]
                    if y_local_idx != -1
                    else -1,
                    "label": y_test_all[key][y_local_idx]
                    if y_local_idx != -1
                    else samples[(key[0], key[1], cell_local_idx)],
                }
                rows_list.append(tmp_dict)
            except Exception as e:
                samp_count += 1
    logging.debug("Samples %s", samp_count)
    results_df = pd.DataFrame(
        rows_list,
        columns=[
            "table_id",
            "table_name",
            "table_shape",
            "col_id",
            "col_name",
            "cell_idx",
            "cell_value",
            "predicted",
            "label",
        ],
    )
    return results_df


def get_results_per_table(result_df):
    logging.debug("Getting results per table")
    results_per_table = dict()
    for table_id in result_df["table_id"].unique():
        table_df = result_df[result_df["table_id"] == table_id]
        table_name = table_df["table_name"].unique()[0]
        table_shape = table_df["table_shape"].unique()[0]
        tp = len(table_df[(table_df["predicted"] == 1) & (table_df["label"] == 1)])
        fp = len(table_df[(table_df["predicted"] == 1) & (table_df["label"] == 0)])
        fn = len(table_df[(table_df["predicted"] == 0) & (table_df["label"] == 1)])
        tn = len(table_df[(table_df["predicted"] == 0) & (table_df["label"] == 0)])
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f_score = (
            (2 * precision * recall) / (precision + recall)
            if precision + recall != 0
            else 0
        )
        results_per_table[table_id] = {
            "table_name": table_name,
            "table_shape": table_shape,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f_score": f_score,
        }
    return results_per_table


def get_tables_dict(init_tables_dict, sandbox_path, dirty_file_names, clean_file_names):
    logging.debug("Getting tables dict")
    all_tables_dict = {}
    table_dirs = os.listdir(sandbox_path)
    table_dirs.sort()
    for table in table_dirs:
        if not table.startswith("."):
            table_path = os.path.join(sandbox_path, table)
            table_file_name_santos = init_tables_dict[table]
            table_id = hashlib.md5(table_file_name_santos.encode()).hexdigest()
            table_df = read_csv(
                os.path.join(table_path, dirty_file_names), low_memory=False
            )
            all_tables_dict[table_id] = {
                "name": table,
                "schema": table_df.columns.tolist(),
                "shape": table_df.shape,
            }
    return all_tables_dict


def get_all_results(
    init_tables_dict,
    tables_path,
    results_dir,
    y_test_all,
    y_local_cell_ids,
    predicted_all,
    y_labeled_by_user_all,
    unique_cells_local_index_collection,
    samples,
    dirty_file_names,
    clean_file_names
):
    logging.info("Getting all results")
    with open(os.path.join(results_dir, "labeled_by_user.pickle"), "wb") as file:
        pickle.dump(y_labeled_by_user_all, file)
    logging.info("Getting classification results")
    
    get_classification_results(
        y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples
    )
    # logging.info("Getting prediction results")
    # tables_dict = get_tables_dict(init_tables_dict, tables_path, dirty_file_names, clean_file_names)
    # results_df = create_predictions_dict(
        # tables_dict,
        # y_test_all,
        # y_local_cell_ids,
        # predicted_all,
        # unique_cells_local_index_collection,
        # samples,
    # )
    # logging.info("Getting results per table")
    # results_per_table = get_results_per_table(results_df)
    # logging.info("Saving results")
    # with open(os.path.join(results_dir, "results_df.pickle"), "wb") as file:
        # pickle.dump(results_df, file)
    # with open(os.path.join(results_dir, "results_per_table.pickle"), "wb") as file:
        # pickle.dump(results_per_table, file)
