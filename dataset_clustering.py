from matplotlib.pyplot import axis
import pandas as pd 
import os
import re
import string

import nltk
import numpy as np
import pandas as pd

from collections import Counter 

from gensim.models import Word2Vec

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from ds_utils.config import set_display_options, set_random_seed
from ds_utils.clustering import Tokenizer, load_data, clean_news_data, vectorize, mbkmeans_clusters

import gensim.downloader as api

def get_column_content(df_col):
    return ' '.join(df_col)

class table:
    def __init__(self, table_id: int, table_path: str) -> None:
        self.table_id = table_id
        self.table_path = table_path
        self.dask_df = pd.read_csv(table_path, header = 0)

    def get_headers(self) -> str:
        return ' '.join(self.dask_df.columns)

    def get_content(self):
        text = self.dask_df.apply(lambda x: get_column_content(x.to_string()), axis = 1)
        return text
        
my_table = table(1, "/Users/fatemehahmadi/Documents/Github-Private/Fatemeh/end-to-end-eds/outputs/raha-datasets/col_groups/col_df_labels_cluster_0.csv")
my_table.get_content()

class lake_details:
    def __init__(self) -> None:
        pass
nltk.download("stopwords")

def get_context_df(sandbox_path):
    context_dict = {'table_id':[], 'parent':[], 'table_name':[],'headers':[], 'content':[]}
    list_dirs_in_snd = os.listdir(sandbox_path)
    table_id = 0
    for parent in list_dirs_in_snd:
        table_dirs_path = os.path.join(sandbox_path, parent)
        table_dirs = os.listdir(table_dirs_path)
        for table in table_dirs:
            context_dict['table_id'].append(table_id)
            context_dict['parent'].append(parent)
            context_dict['table_name'].append(table)
            context_dict['headers'].append(get_df_headers(os.path.join(table_dirs_path, table + "/dirty.csv")))

            df_path = os.path.join(table_dirs_path, table + "/dirty.csv")
            df_table = pd.read_csv(df_path)
            df_table = df_table.select_dtypes(include=object)
            df_table_text = ""
            for i in df_table.columns:
                col_text = ' '.join(df_table[i].astype(str).tolist())
                df_table_text += col_text
            context_dict['content'].append(df_table_text)
            table_id += 1
    context_df = pd.DataFrame.from_dict(context_dict)
    return context_df

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
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords
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

def cluster_datasets(sandbox_path, output_path):
    custom_stopwords = set(stopwords.words("english"))
    context_df = get_context_df(sandbox_path)
    text_columns = ["parent", "table_name", "headers", "content"]

    df = context_df.copy()
    df = df.fillna("")

    for col in text_columns:
        df[col] = df[col].astype(str)

    # Create text column based on parent, table_name, and headers
    df["text"] = df[text_columns].apply(lambda x: " | ".join(x), axis=1)
    df["tokens"] = df["text"].map(lambda x: clean_text(x, word_tokenize, custom_stopwords))

    # Remove duplicated after preprocessing
    _, idx = np.unique(df["tokens"], return_index=True)
    df = df.iloc[idx, :]

    # Remove empty values
    df = df.loc[df.tokens.map(lambda x: len(x) > 0), ["text", "tokens"]]

    print(f"Original dataframe: {context_df.shape}")
    print(f"Pre-processed dataframe: {df.shape}")

    docs = df["text"].values
    tokenized_docs = df["tokens"].values
    vocab = Counter()
    for token in tokenized_docs:
        vocab.update(token)

    print(vocab.most_common(10))

    k = 1
    if k > 1:
        #model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1, seed=42)
        wv = api.load('word2vec-google-news-300')
        vectorized_docs = vectorize(tokenized_docs, model=wv)
        print(len(vectorized_docs), len(vectorized_docs[0]))
        clustering, cluster_labels = mbkmeans_clusters(X=vectorized_docs, k=k, print_silhouette_values=False)
        print("Top terms per cluster (based on centroids):")
        for i in range(3):
            tokens_per_cluster = ""
            most_representative = wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
            for t in most_representative:
                tokens_per_cluster += f"{t[0]} "
            print(f"Cluster {i}: {tokens_per_cluster}")
    else:
        cluster_labels = np.zeros(len(tokenized_docs)).tolist()
        df_clusters = pd.DataFrame({
            "text": docs,
            "tokens": [" ".join(text) for text in tokenized_docs],
            "cluster": cluster_labels
        })
        print(df_clusters)

        context_df = context_df.join(df_clusters)
        context_df.to_csv(output_path)

