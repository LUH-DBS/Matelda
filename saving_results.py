import pickle
import os
import pandas as pd
import app_logger

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

logger = app_logger.get_logger()


def save_results(n_samples, y_test, predicted, tables_dict, results_path, original_data_values):
    classifier_results = []
    for i in range(len(predicted)):
        classifier_results.append((predicted[i], int(y_test[i])))

    results_df = create_df(original_data_values, classifier_results, tables_dict)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, "results_df.pickle"), "wb") as file:
        pickle.dump(results_df, file)

    logger.info(classification_report(y_true=y_test, y_pred=predicted))
    tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=predicted).ravel()
    precision, recall, f_score, support = precision_recall_fscore_support(y_test, predicted, average='macro')
    logger.info("tn: {}, fp: {}, fn: {}, tp: {}".format(tn, fp, fn, tp))
    scores = {"n_samples": n_samples, "precision": precision, "recall": recall, "f_score": f_score, "support": support, "tp": tp, "fp": fp,
              "fn": fn, "tn": tn}
    with open(os.path.join(results_path, "scores.pickle"), "wb") as file:
        pickle.dump(scores, file)

    return


def create_df(original_data_values, classification_results, all_tables_dict):
    rows_list = []
    for idx_, i in enumerate(original_data_values):
        try:
            tmp_dict = {'table_id': i[0], 'table_name': all_tables_dict[i[0]]['name'], 'col_id': i[1],
                        'col_name': all_tables_dict[i[0]]['schema'][i[1]], 'cell_idx': i[2], 'cell_value': i[3],
                        'predicted': classification_results[idx_][0], 'label': classification_results[idx_][1]}
            rows_list.append(tmp_dict)
        except Exception as e:
            print(e)
            print(i)
    results_df = pd.DataFrame(rows_list,
                              columns=['table_id', 'table_name', 'col_id', 'col_name', 'cell_idx', 'cell_value',
                                       'predicted', 'label'])
    return results_df


def get_tables_dict(sandbox_path):
    all_tables_dict = {}
    table_id = 0
    table_dirs = os.listdir(sandbox_path)
    for table in table_dirs:
        table_path = os.path.join(sandbox_path, table)
        table_df = pd.read_csv(table_path + "/dirty.csv")
        all_tables_dict[table_id] = {"name": table, "schema": table_df.columns.tolist()}
        table_id += 1
    return all_tables_dict


# TODO
def get_all_results(tables_path, results_dir, original_data_values, n_samples,
                    y_test, predicted):
    tables_dict = get_tables_dict(tables_path)
    save_results(n_samples, y_test, predicted, tables_dict, results_dir, original_data_values)

