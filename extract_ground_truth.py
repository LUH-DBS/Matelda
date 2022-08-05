from codecs import ignore_errors
import os
import pickle
import pandas as pd 

def generate_labels(dirty_df, clean_df):
    dirty_df = dirty_df.fillna(0)
    clean_df = clean_df.fillna(0)
    dirty_df.columns = clean_df.columns
    for dcol in dirty_df:
        try:
            dirty_df[dcol] = dirty_df[dcol].astype(clean_df[dcol].dtype)
        except Exception as e:
            print(e)
            dirty_df[dcol] = dirty_df[dcol].astype(str)
            clean_df[dcol] = clean_df[dcol].astype(str)
    label_df = clean_df.eq(dirty_df)
    return label_df*1

def extract_gt(sand_path, output_path):
    table_id = 0
    path = os.listdir(sand_path)
    datasets_gt = dict()
    for p in path:
        parent = os.path.join(sand_path, p)
        for table in os.listdir(parent):
            try:
                dirty_df = pd.read_csv(os.path.join(parent, table) + "/dirty.csv")
                clean_df = pd.read_csv(os.path.join(parent, table) + "/" + table + ".csv")
                label_df = generate_labels(dirty_df, clean_df)
                for col_id in range(len(label_df.columns)):
                    datasets_gt[(table_id, col_id)] = label_df[label_df.columns[col_id]]
                table_id += 1                
            except Exception as e:
                print(e)
                table_id += 1             
    filehandler = open(output_path,"wb")
    pickle.dump(datasets_gt, filehandler)
    filehandler.close()

# filehandler = open("/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/MVP/Classification/error-detection-at-sacle/classification/gt.pickle","rb")
# dgt = pickle.load(filehandler)
# filehandler.close()
