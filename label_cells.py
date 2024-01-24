import os
import pandas as pd

def label(sampling_df_path, manual_labels_base_path):
    sampling_df = pd.read_csv(sampling_df_path)
    labeled = 0
    for i, row in sampling_df.iterrows():
        path = os.path.join(manual_labels_base_path, f"{row['table_name']}".removesuffix(".csv"), f"raha-baran-results-{row['table_name']}".removesuffix(".csv"), "labled_values.csv")
        labeled_values = pd.read_csv(path)
        try:
            labeled_values = labeled_values[(labeled_values["Column"] == row["col_idx"]) & (labeled_values["Row"] == row["row_idx"]) & (labeled_values["Value"] == row["cell_value"])]
            if len(labeled_values) == 0:
                raise Exception("label not found")
            else:
                sampling_df.loc[i, "label"] = labeled_values["Label"].values[0]
                labeled += 1
                print(labeled_values)
        except Exception as e:
            # print(e)
            # print(f"label not found - table: {row['table_name']}")
            continue

    print(f"labeled: {labeled}")
    sampling_df.to_csv(os.path.join(os.path.dirname(sampling_df_path), "labeled_sampling_df.csv"), index=False)

label("/home/fatemeh/VLDB-Jan-Manual-Exp/ED-Scale/output_quintet_1/sampling_df.csv", 
      "/home/fatemeh/VLDB-Jan-Manual-Exp/ED-Scale/output_quintet_1/_spell_checker_Quintet_66_labels")