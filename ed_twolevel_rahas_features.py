from collections import Counter
from multiprocessing import freeze_support
import os
import pickle
import random
import warnings
import numpy as np
import pandas as pd
from pip import main
from scipy import rand
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import generate_raha_features
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from imblearn.over_sampling import RandomOverSampler


def get_cells_features(sandbox_path, outputpath):
    features_dict = dict()
    table_id = 0
    list_dirs_in_snd = os.listdir(sandbox_path)
    for parent in list_dirs_in_snd:
        table_dirs_path = os.path.join(sandbox_path, parent)
        table_dirs = os.listdir(table_dirs_path)
        for table in table_dirs:
            try:
                path = os.path.join(table_dirs_path, table)
                col_features = np.asarray(generate_raha_features.generate_raha_features(table_dirs_path, table))
                for col_idx in range(len(col_features)):
                    for row_idx in range(len(col_features[col_idx])):
                        features_dict[(table_id, col_idx, row_idx, 'og')] = col_features[col_idx][row_idx]
                clean_df = pd.read_csv(path + "/" + table + ".csv")
                dirty_df = pd.read_csv(path + "/" + "dirty.csv")
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
                for col_idx, col_name in enumerate(label_df.columns):
                    for row_idx in range(len(label_df[col_name])):
                        features_dict[(table_id, col_idx, row_idx, 'gt')] = label_df[col_name][row_idx]
                table_id += 1
                print(table_id)
            except Exception as e:
                print(e)
    filehandler = open(os.path.join(outputpath, "features.pkl"),"wb")
    pickle.dump(features_dict,filehandler)
    filehandler.close()
    return features_dict

def classify(X_train, y_train, X_test, y_test):
    imp = SimpleImputer(strategy="most_frequent")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                            max_depth=1, random_state=0)
    clf = make_pipeline(imp, gbc)
    clf.fit(np.asarray(X_train), np.asarray(y_train))
    predicted = clf.predict(X_test)
    precision, recall, f_score, support = score(y_test, predicted, average='macro')
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
    print("classification_report:", classification_report(y_test, predicted))
    print("confusion_matrix ", tn, fp, fn, tp)
    print("scores ", precision, recall, f_score)

    return tp, tn, fp, fn, precision, recall, f_score

def get_cols(col_groups_files_path, features_dict):
    tp_all, tn_all, fp_all, fn_all = 0, 0, 0, 0
    X_train = []
    y_train = []
    X_test = []
    y_test = []


    for file in os.listdir(col_groups_files_path):
        if ".pickle" in file:
            file = open(os.path.join(col_groups_files_path, file),'rb')
            group_df = pickle.load(file)
            file.close()
            clusters = set(group_df['col_cluster_label'].sort_values())
            for c in clusters:
                try:
                    c_df = group_df[group_df['col_cluster_label'] == c]
                    X = []
                    y = []
                    for index, row in c_df.iterrows():
                        for cell_idx in range(len(row['col_value'])):
                            X.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'og')].tolist())
                            y.append(features_dict[(row['table_id'], row['col_id'], cell_idx, 'gt')].tolist())
                                            
                    n_clusters = 100
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                    cells_per_cluster = dict()
                    labels_per_cluster = dict()
                    for cell in enumerate(kmeans.labels_):
                        if cell[1] in cells_per_cluster.keys():
                            cells_per_cluster[cell[1]].append(cell[0])
                        else:
                            cells_per_cluster[cell[1]] = [cell[0]]
                    
                    for key in cells_per_cluster.keys():
                        labels_tmp = []
                        for n in range(n_clusters):
                            sample = random.choice(cells_per_cluster[key])
                            label = y[sample]
                            labels_tmp.append(label)
                        z_c = labels_tmp.count(0)
                        o_c = labels_tmp.count(1)
                        if o_c >= z_c:
                            labels_per_cluster[key] = 1
                        else:
                            labels_per_cluster[key] = 0


                    for key in list(cells_per_cluster.keys())[0:len(cells_per_cluster.keys())-1]:
                        for cell in cells_per_cluster[key]:
                            X_train.append(X[cell])
                            y_train.append(labels_per_cluster[key])

                    c = list(cells_per_cluster.keys())[-1]
                    for cell in cells_per_cluster[c]:
                        X_test.append(X[cell])
                        y_test.append(y[cell])
                except Exception as e:
                    print(e)
    
    classify(X_train, y_train, X, y)
    return X, y

if __name__== '__main__':
    sandbox_path = "/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/MVP/raha-datasets"
    output_path = "/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/end-to-end-eds/outputs/raha-datasets"
    col_groups_path = os.path.join(output_path, "col_groups")

    freeze_support()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # features_dict = get_cells_features(sandbox_path, output_path)
        file = open(os.path.join(output_path, "features.pkl"), 'rb')
        features_dict = pickle.load(file)
        file.close()
        get_cols(col_groups_path, features_dict)
        print("")

