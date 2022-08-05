from cmath import nan
from collections import Counter
import math
import os
import pickle
import profile
import sys
import numpy as np
import pandas as pd
from openclean.profiling.dataset import dataset_profile
from sklearn.cluster import DBSCAN, OPTICS
from dataset_clustering import clean_text
from ds_utils import clustering
from nltk import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
import nltk
from gensim.models import Word2Vec
import dask.dataframe as dd


nltk.download("stopwords")

type_dicts = {'int':0, 'float':1, 'str':2, 'date':3}

def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features




def get_clusters_dict(df, n_clusters):
    clusters_dict = {}

    for i in range(n_clusters):
        clusters_dict[i] = []

    row_iterator = df.iterrows()
    for i, row in row_iterator:
        clusters_dict[row['cluster']].append((row['table_id'], row['parent'], row['table_name']))
    return clusters_dict


def get_col_df(snd_path, cluster):
    col_dict = {'table_id':[], 'col_id':[], 'col_value':[], 'col_gt':[], 'col_type':[]}
    dgt = extract_gt("/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/end-to-end-eds/outputs/raha-datasets/gt.pickle")
    for table in cluster:
        parent_path = os.path.join(snd_path, table[1])
        table_path = os.path.join(parent_path, table[2] + "/dirty.csv")
        df = pd.read_csv(table_path)
        for col in df.columns:
            col_dict['table_id'].append(table[0])
            col_dict['col_id'].append(df.columns.tolist().index(col))
            col_dict['col_value'].append(df[col].tolist())
            col_dict['col_gt'].append(dgt[(table[0], df.columns.tolist().index(col))].values)
            col_dict['col_type'].append(df[col].dtype)
    col_df = pd.DataFrame.from_dict(col_dict)
    return col_df

def get_col_features(col_df):
    # custom_stopwords = set(stopwords.words("english"))
    # wv = api.load('word2vec-google-news-300')
    
    col_features = []
    print(col_df.shape[0])
    characters_dictionary = {}
    values_dictionary = {}

    for i in range(col_df.shape[0]):
        print(i)
        features = []
        features.append(col_df['table_id'][i])
        features.append(col_df['col_id'][i])
        profiles = dataset_profile(pd.DataFrame(col_df['col_value'][i]))
        features.append(profiles[0]['stats']['totalValueCount'])
        features.append(profiles[0]['stats']['emptyValueCount'])
        features.append(profiles[0]['stats']['distinctValueCount'])
        features.append(profiles.stats()['uniqueness'][0])
        features.append(profiles[0]['stats']['entropy'])

        try:
            col_type = profiles.types().columns[0]
            features.append(type_dicts[col_type])
        except Exception as e:
            features.append(-1)
        for j in range(len(features)):
            if features[j] == None:
                features[j] = -1
        col_features.append(features)

        for value in col_df['col_value'][i]:
            for character in list(set(list(str(value)))):
                if character not in characters_dictionary:
                    characters_dictionary[character] = 0.0
                characters_dictionary[character] += 1.0
            if value not in values_dictionary: 
                values_dictionary[value] = 0.0
            values_dictionary[value] += 1.0

    for i in range(col_df.shape[0]):
        column_profile = {
            "characters": {ch: characters_dictionary[ch] / len(col_df['col_value'][i]) for ch in characters_dictionary},
            "values": {v: values_dictionary[v] / len(col_df['col_value'][i]) for v in values_dictionary},
        }
        char_list = list(column_profile["characters"].values())
        value_list = list(column_profile["values"].values())
        for char in char_list:
            col_features[i].append(char)
        for val in value_list:
            col_features[i].append(val)

    return col_features

def cluster_cols_auto(col_features):
    reduced_features = []
    for col_feature in col_features:
        reduced_features.append(col_feature[6:])
    clustering = DBSCAN(eps=3, min_samples=2).fit(reduced_features)
    columns=['table_id', 'col_id', 'totalValueCount', 'emptyValueCount', 'distinctValueCount', 'uniqueness', 'entropy', 'dtype_code']
    voc = [str(i) for i in range(8, len(col_features[0]))]
    columns = columns + voc
    col_labels_df = pd.DataFrame(col_features, columns=columns)
    col_labels_df['col_cluster_label'] = pd.DataFrame(clustering.labels_)
    return col_labels_df

def extract_gt(gt_path):
    filehandler = open(gt_path,"rb")
    dgt = pickle.load(filehandler)
    filehandler.close()
    return dgt


def col_folding(context_df_path, sandbox_path, output_path):
    df = dd.read_csv(context_df_path)
    clusters_dict = get_clusters_dict(df, n_clusters=1)
    try:
        os.mkdir(os.path.join(output_path, "col_groups"))
    except OSError as error:
        print(error) 

    for cluster in clusters_dict:
        col_df = get_col_df(sandbox_path, clusters_dict[cluster])
        col_features = get_col_features(col_df)
        col_labels_df = cluster_cols_auto(col_features)
        col_labels_df['col_value'] = col_df['col_value']
        col_labels_df['col_gt'] = col_df['col_gt']
        filehandler = open(os.path.join(output_path, "col_groups/col_df_labels_cluster_{}.pickle".format(cluster)),"wb")
        pickle.dump(col_labels_df, filehandler)
        filehandler.close()
        col_labels_df[["col_cluster_label", "col_value"]].to_csv(os.path.join(output_path, "col_groups/col_df_labels_cluster_{}.csv".format(cluster)))
