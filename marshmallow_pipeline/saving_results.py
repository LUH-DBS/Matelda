from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import os
import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix

from marshmallow_pipeline.utils.read_data import read_csv


def get_classification_results(
    y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples, unique_cells_local_ids_collection
):
    logging.debug("Classification results:")
    total_tn, total_fp, total_fn, total_tp = 0, 0, 0, 0
    try:
        for i in predicted_all.keys():
            cell_local_ids = unique_cells_local_ids_collection[i]
            swapped_cell_local_ids = {v: (k[0], k[1], k[2]) for k, v in cell_local_ids.items()}
            col_cluster_prediction = list(predicted_all[i])
            for j in range(len(col_cluster_prediction)):
                if swapped_cell_local_ids[j] in samples:
                    col_cluster_prediction[j] = samples[swapped_cell_local_ids[j]]
            col_cluster_y = y_test_all[i]

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

        total_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else None
        total_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else None
        total_fscore = (2 * total_precision * total_recall) / (
            total_precision + total_recall
        ) if total_precision and total_recall else None
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
    except Exception as e:
        logging.error("Error: %s", e)

def process_cell_cell_results(cell_key, key, unique_cells_local_index_collection, y_local_cell_ids, y_test_all, predicted_all, all_tables_dict, samples):
    try:
        # print(cell_key)
        cell_local_idx = unique_cells_local_index_collection[key][cell_key]
        y_cell_ids = {id: idx for idx, id in enumerate(y_local_cell_ids[key])}
        y_local_idx = y_cell_ids.get(cell_local_idx, -1)
        res_dict = {
            "table_id": cell_key[0],
            "table_name": all_tables_dict[cell_key[0]]["name"],
            "table_shape": all_tables_dict[cell_key[0]]["shape"],
            "col_id": cell_key[1],
            "col_name": all_tables_dict[cell_key[0]]["schema"][cell_key[1]],
            "cell_idx": cell_key[2],
            "cell_value": cell_key[3],
            "predicted": predicted_all[key][y_local_idx],
            "label": y_test_all[key][y_local_idx]
        }
    except Exception as e:
        logging.error("Error: %s", e)
        return None
    print("key - done: ", key)
    return res_dict

def proccess_col_group_cell_results(key, unique_cells_local_index_collection, y_local_cell_ids, y_test_all, predicted_all, all_tables_dict, samples, executor):
    print(key)
    print(len(unique_cells_local_index_collection[key]))
    futures = []
    for cell_key in unique_cells_local_index_collection[key]:
        future = executor.submit(process_cell_cell_results, \
                                    cell_key, key, unique_cells_local_index_collection,\
                                        y_local_cell_ids, y_test_all, predicted_all,\
                                            all_tables_dict, samples)
        futures.append(future)
    return futures

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
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for key in unique_cells_local_index_collection:
            future = executor.submit(proccess_col_group_cell_results, key,
                                    unique_cells_local_index_collection,
                                    y_local_cell_ids, y_test_all, predicted_all,
                                    all_tables_dict, samples, executor)
            futures.append(future)
    
        # Wait for all the tasks to complete
        for future in futures:
            rows_list.append(res_dict for res_dict in future.result())

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
                os.path.join(table_path, dirty_file_names), low_memory=False, data_type='str'
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
        y_test_all, predicted_all, y_labeled_by_user_all, results_dir, samples, unique_cells_local_index_collection
    )
    logging.info("Getting prediction results")
    # tables_dict = get_tables_dict(init_tables_dict, tables_path, dirty_file_names, clean_file_names)
    # results_df = create_predictions_dict(
    #     tables_dict,
    #     y_test_all,
    #     y_local_cell_ids,
    #     predicted_all,
    #     unique_cells_local_index_collection,
    #     samples,
    # )
    # logging.info("Getting results per table")
    # results_per_table = get_results_per_table(results_df)
    # logging.info("Saving results")
    # with open(os.path.join(results_dir, "results_df.pickle"), "wb") as file:
    #     pickle.dump(results_df, file)
    # with open(os.path.join(results_dir, "results_per_table.pickle"), "wb") as file:
    #     pickle.dump(results_per_table, file)


def get_all_results_from_disk(output_path, tables_path, dirty_file_names, clean_file_names):
    logging.info("Getting all results from disk")
    with open(os.path.join(output_path, "tables_dict.pickle"), "rb") as file:
        tables_init_dict = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "y_test_all.pickle"), "rb") as file:
        y_test_all = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "y_local_cell_ids.pickle"), "rb") as file:
        y_local_cell_ids = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "predicted_all.pickle"), "rb") as file:
        predicted_all = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "y_labeled_by_user_all.pickle"), "rb") as file:
        y_labeled_by_user_all = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "unique_cells_local_index_collection.pickle"), "rb") as file:
        unique_cells_local_index_collection = pickle.load(file)
    with open(os.path.join(output_path, "results", "final_results", "samples.pickle"), "rb") as file:
        samples = pickle.load(file)
    get_all_results(
        tables_init_dict,
        tables_path,
        os.path.join(output_path, "results"),
        y_test_all,
        y_local_cell_ids,
        predicted_all,
        y_labeled_by_user_all,
        unique_cells_local_index_collection,
        samples,
        dirty_file_names,
        clean_file_names
    )    

# output_path = "/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/output-test-quintet/_otg_Quintet_7_labels"
# tables_path = "/home/fatemeh/ED-Scale-mp-dgov/ED-Scale/data/Quintet"
# dirty_file_names = "dirty.csv"
# clean_file_names = "clean.csv"

# get_all_results_from_disk(output_path, tables_path, dirty_file_names, clean_file_names)
