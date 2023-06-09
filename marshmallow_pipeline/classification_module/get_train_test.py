import logging

import pandas as pd

def get_train_test_sets(X_temp, y_temp, samples_dict, cell_clustering_df):
    logging.info("Train-Test set preparation")
    cells_per_cluster = cell_clustering_df["cells_per_cluster"].values[0]
    samples_df = pd.DataFrame(samples_dict)
    X_train, y_train, X_test, y_test, y_cell_ids = [], [], [], [], []
    clusters = samples_df["cell_cluster"].unique().tolist()
    clusters.sort()
    for key in clusters:
        try:
            if key == -1:
                continue
            cell_cluster_samples = samples_df[samples_df["cell_cluster"] == key]["samples_indices_global"].values[0]
            cell_cluster_final_label = samples_df[samples_df["cell_cluster"] == key]["final_label_to_be_propagated"].values[0]
            if len(cell_cluster_samples) == 0:
                for cell in cells_per_cluster[key]:
                    X_test.append(X_temp[cell])
                    y_test.append(y_temp[cell])
                    y_cell_ids.append(cell)
            else:         
                for cell in cells_per_cluster[key]:
                    X_train.append(X_temp[cell])
                    if cell in cell_cluster_samples:
                        y_train.append(y_temp[cell])
                    else:
                        y_train.append(cell_cluster_final_label)
                        X_test.append(X_temp[cell])
                        y_test.append(y_temp[cell])
                        y_cell_ids.append(cell)
        except Exception as e:
            logging.error("Error in get_train_test_sets: %s", e)

    logging.info("Length of X_train: %s", len(X_train))
    return X_train, y_train, X_test, y_test, y_cell_ids
