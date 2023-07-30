import hashlib
import os
import pickle 
import pandas as pd

def get_samples_dict_cell_cluster(path):
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    return samples

def get_hashed_table_names(tables_dict_path, ):
    with open(tables_dict_path, "rb") as f:
        tables_dict = pickle.load(f)
    tables_hash_dict = {}
    for k, v in tables_dict.items():
        hash_name = str(hashlib.md5(v.encode()).hexdigest())
        tables_hash_dict[hash_name] = v
    return tables_hash_dict

def get_sampling_df(samples, tables_hash_dict, table_cluster, col_cluster):
    sampling_df = pd.DataFrame(columns=['table_cluster', 'col_cluster', 'cell_cluster', 'table_name', 'col_idx', 'row_idx', 'cell_value', 'label', 'location_in_dict_i', 'location_in_dict_j'])
    samples_swapped = {v: k for k, v in samples['datacell_uids'].items()}
    samples_global_idx = samples["samples_indices_global"]
    for i in range(len(samples_global_idx)):
        for j in range(len(samples_global_idx[i])):
            uid = samples_swapped[samples_global_idx[i][j]]
            sampling_df.loc[len(sampling_df)] = [
                table_cluster,
                col_cluster,
                samples['cell_cluster'][i],
                tables_hash_dict[uid[0]],
                uid[1],
                uid[2],
                uid[3],
                samples['labels'][i][j],
                i,
                j]
    return sampling_df


path = "/home/fatemeh/EDS-Precision-Exp/ED-Scale/output-precision/_precision-exp_wdc-sampled_378_labels/samples_dict/samples_dict_0_0.pkl"
tables_dict_path = "/home/fatemeh/EDS-Precision-Exp/ED-Scale/output-precision/_precision-exp_wdc-sampled_378_labels/tables_dict.pickle"
base_path = "/home/fatemeh/EDS-Precision-Exp/ED-Scale/output-precision/_precision-exp_wdc-sampled_378_labels/samples_dict"
tables_hash_dict = get_hashed_table_names(tables_dict_path)
sampling_results = []
for path in os.listdir(base_path):
    if path.endswith(".pkl"):
        samples = get_samples_dict_cell_cluster(os.path.join(base_path, path))
        tabcolclup = path.removeprefix("samples_dict_").removesuffix(".pkl")
        table_cluster = int(tabcolclup.split("_")[0])
        col_cluster = int(tabcolclup.split("_")[1])
        sampling_results.append(get_sampling_df(samples, tables_hash_dict, table_cluster, col_cluster))

sampling_df = pd.concat(sampling_results)
sampling_df.to_csv("sampling_df.csv")
print(sampling_df)
