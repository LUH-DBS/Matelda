import logging

from matplotlib.pyplot import axis
import pandas as pd 
import os
import re
import string

import nltk
import numpy as np
import pandas as pd

from collections import Counter 

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score

import app_logger
from ds_utils.config import set_display_options, set_random_seed
from ds_utils.clustering import Tokenizer, load_data, clean_news_data, vectorize, mbkmeans_clusters

import gensim.downloader as api


def get_column_content(df_col):
    return ' '.join(df_col)

nltk.download("stopwords")


def get_df_headers(table_path):
    df = pd.read_csv(table_path)
    return ' '.join(df.columns)


def get_context_df(sandbox_path):
    context_dict = {'table_id': [], 'parent': [], 'table_name': [], 'headers': [], 'content': []}
    sandbox_children = os.listdir(sandbox_path)
    sandbox_children.sort()
    table_id = 0
    total_num_cells = 0
    for child_name in sandbox_children:
        if not child_name.startswith("."):
            child_path = os.path.join(sandbox_path, child_name)
            tables_dirs = os.listdir(child_path)
            tables_dirs.sort()
            for table in tables_dirs:
                if not table.startswith("."):
                    table_path = os.path.join(child_path, table)
                    df_path = os.path.join(table_path, "dirty_clean.csv")
                    df_text_columns = pd.read_csv(df_path)
                    total_num_cells += df_text_columns.size
                    df_text_columns = df_text_columns.select_dtypes(include=object)
                    df_table_text = ""
                    for column in df_text_columns.columns:
                        col_text = ' '.join(df_text_columns[column].astype(str).tolist())
                        df_table_text += col_text

                    context_dict['table_id'].append(table_id)
                    context_dict['parent'].append(child_name)
                    context_dict['table_name'].append(table)
                    context_dict['headers'].append(get_df_headers(os.path.join(table_path, "dirty_clean.csv")))
                    context_dict['content'].append(df_table_text)
                    table_id += 1
    context_df = pd.DataFrame.from_dict(context_dict)
    return context_df, total_num_cells


def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = ''.join(word.strip(string.punctuation) for word in text)
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if t not in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens


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
            if token in model:
                try:
                    vectors.append(model[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def cluster_datasets(sandbox_path, output_path, auto_clustering_enabled):
    logger = logging.getLogger()

    custom_stopwords = set(stopwords.words("english"))
    context_df, total_num_cells = get_context_df(sandbox_path)

    df = context_df.copy()
    df = df.fillna("")

    text_columns = ["parent", "table_name", "headers", "content"]
    for col in text_columns:
        df[col] = df[col].astype(str)

    # Create text column based on parent, table_name, and headers
    # TODO: check licence
    df["text"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
    df["tokens"] = df["text"].map(lambda x: clean_text(x, word_tokenize, custom_stopwords))

    # Remove duplicated after preprocessing
    _, idx = np.unique(df["tokens"], return_index=True)
    df = df.iloc[idx, :]

    # Remove empty values
    df = df.loc[df.tokens.map(lambda x: len(x) > 0), ["text", "tokens"]]

    logger.info(f"Original dataframe: {context_df.shape}")
    logger.info(f"Pre-processed dataframe: {df.shape}")

    docs = df["text"].values
    tokenized_docs = df["tokens"].values
    vocab = Counter()
    for token in tokenized_docs:
        vocab.update(token)

    logger.info(f"Most common vocabs are: {vocab.most_common(10)}")

    if auto_clustering_enabled:
        # TODO: embedding model and DBSCAN params in config file
        # model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1, seed=42)
        model = api.load('word2vec-google-news-300')
        vectorized_docs = vectorize(tokenized_docs, model=model)
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(vectorized_docs)
        cluster_labels = clustering.labels_
    else:
        cluster_labels = np.ones(len(tokenized_docs)).tolist()

    df_clusters = pd.DataFrame({
        "text": docs,
        "tokens": [" ".join(text) for text in tokenized_docs],
        "cluster": cluster_labels
    })
    num_clusters = len(set(cluster_labels))

    # TODO: Change join
    context_df = context_df.join(df_clusters)
    context_df.to_csv(output_path)

    return context_df, num_clusters, total_num_cells

