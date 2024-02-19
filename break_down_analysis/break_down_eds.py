import pandas as pd
import pickle 

dataset_path = "/home/fatemeh/VLDB-Jan/ED-Scale-Main/ED-Scale/datasets/Quintet_Breakdown"
movies_1_err_df = pd.read_csv(f"{dataset_path}/movies_1/error_types.csv")
hospital_err_df = pd.read_csv(f"{dataset_path}/hospital/error_types.csv")
beers_err_df = pd.read_csv(f"{dataset_path}/beers/error_types.csv")
rayyan_err_df = pd.read_csv(f"{dataset_path}/rayyan/error_types.csv")
flights_err_df = pd.read_csv(f"{dataset_path}/flights/error_types.csv")

for exec in range(5):
    for labeling_budget in [7, 16, 33, 50, 66, 132, 198, 330, 660, 990, 1320]:
        print(f"Processing {exec}th execution with {labeling_budget} labels")
        path = f"/home/fatemeh/VLDB-Jan/ED-Scale-Main/ED-Scale/experiments/results/EDS_Standard/output_Quintet/output_Quintet_{exec}/_vldb_march_Quintet_{labeling_budget}_labels/results/results_df.pickle"
        res_path = f"/home/fatemeh/VLDB-Jan/ED-Scale-Main/ED-Scale/experiments/results/EDS_Standard/output_Quintet/output_Quintet_{exec}/_vldb_march_Quintet_{labeling_budget}_labels/results/results_df_with_error_type.csv"

        with open(path, 'rb') as handle:
            res_df = pickle.load(handle)

        res_df = pd.DataFrame(res_df)
        res_df["Error Type"] = ""

        for i, row in res_df.iterrows():
            if row["table_name"] == "movies_1":
                error_types_df = movies_1_err_df
            elif row["table_name"] == "hospital":
                error_types_df = hospital_err_df
            elif row["table_name"] == "beers":
                error_types_df = beers_err_df
            elif row["table_name"] == "rayyan":
                error_types_df = rayyan_err_df
            elif row["table_name"] == "flights":
                error_types_df = flights_err_df
            else:
                print("Table name not found")
                break
            res_df.loc[i, "Error Type"]= error_types_df[row["col_name"]][row["cell_idx"]]


        res_df.to_csv(res_path, index=False)