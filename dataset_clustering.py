import pandas as pd
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

import gensim.downloader as api

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import lit, col

# import dbscan
# from scipy.spatial import distance


def clean_text(text, tokenizer, stopwords):
    """Pre-process text and generate tokens

    Args:
        text: Text to tokenize.

    Returns:
        Tokenized text.
    """
    text = "".join(word.strip(string.punctuation) for word in text)
    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation

    tokens = tokenizer(text)  # Get tokens from text
    tokens = [t for t in tokens if t not in stopwords]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
    return tokens


def vectorize(tokens, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    print(type(tokens.collect()))
    print(tokens.collect())
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
        return avg_vec
    else:
        return zero_vector


def cluster_datasets_pyspark(
    csv_paths_df: DataFrame,
    output_path: str,
    table_grouping_enabled: bool,
    auto_clustering_enabled: bool,
):
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    nltk.download("stopwords")

    if table_grouping_enabled:
        logger.warn("Creating context DataFrame")
        context_rdd = csv_paths_df.rdd.map(lambda row: create_table_context(row))
        context_df = context_rdd.toDF(
            ["table_id", "parent", "table_name", "headers", "content", "text", "token"]
        )
        logger.warn("Clustering context DataFrame")
        if auto_clustering_enabled == True:
            logger.warn("Clustering with AUTO_CLUSTERING")
            # TODO: embedding model and DBSCAN params in config file
            # model = Word2Vec(sentences=tokenized_docs, vector_size=100, workers=1, seed=42)
            # model = api.load('word2vec-google-news-300')
            # vectorized_docs_df = context_df.transform(lambda row: vectorize(row.token, model))
            # vectorized_docs_df.show()
            # print(dbscan.process(spark, vectorized_docs_rdd, .5, 5, distance.euclidean, , "checkpoint"))
            # clustering = DBSCAN(eps=0.5, min_samples=5).fit(vectorized_docs)
            # cluster_labels = clustering.labels_
        else:
            logger.warn("Clustering without AUTO_CLUSTERING")
            context_df = context_df.withColumn("cluster", lit(1))

        table_grouping_df = context_df.select(col("table_id"), col("cluster"))
        logger.warn("Writing table clustering result to disk.")
        table_grouping_df.write.parquet(output_path, mode="overwrite")
        table_grouping_df.show()
    else:
        logger.warn("Loading table grouping from disk")
        table_grouping_df = spark.read.parquet(output_path)
    return table_grouping_df


def create_table_context(row):
    custom_stopwords = set(stopwords.words("english"))
    dirty_df = pd.read_csv(
        row.dirty_path,
        sep=",",
        header="infer",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    df_text_columns = dirty_df.select_dtypes(include=object)

    # Table content
    df_table_text = ""
    for column in df_text_columns.columns:
        col_text = " ".join(df_text_columns[column].astype(str).tolist())
        df_table_text += col_text

    # Column names
    df_column_text = " ".join(dirty_df.columns)

    # Create text column based on parent, table_name, and headers
    # TODO: check licence
    text = " | ".join([row.parent, row.table_name, df_column_text, df_table_text])
    tokens = set(clean_text(text, word_tokenize, custom_stopwords))

    # Remove duplicated after preprocessing
    tokens = set(tokens)

    # Remove empty values
    tokens = list(filter(lambda token: len(token) > 0, tokens))

    return [
        row.table_id,
        row.parent,
        row.table_name,
        df_column_text,
        df_table_text,
        text,
        tokens,
    ]
