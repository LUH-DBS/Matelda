import re
import string
from typing import List

import gensim.downloader as api
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import Row
from sklearn.cluster import DBSCAN


def clean_text(text, tokenizer, stopwords) -> List[str]:
    """Pre-process text and generate tokens

    Args:
        text (_type_): Text to tokenize.
        tokenizer (_type_): _description_
        stopwords (_type_): _description_

    Returns:
        List[str]: Tokenized text.
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
        tokens (_type_): _description_
        model (_type_): Gensim's Word Embedding

    Returns:
        _type_: _description_
    """
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
    table_grouping_enabled: int,
    auto_clustering_enabled: int,
) -> DataFrame:
    """_summary_

    Args:
        csv_paths_df (DataFrame): _description_
        output_path (str): _description_
        table_grouping_enabled (int): _description_
        auto_clustering_enabled (int): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    nltk.download("stopwords")

    if table_grouping_enabled == 1:
        logger.warn("Creating context DataFrame")
        context_rdd = csv_paths_df.rdd.map(lambda row: create_table_context(row))
        context_df = context_rdd.toDF(
            #    ["table_id", "parent", "table_name", "headers", "content", "text", "token"]
            ["table_id", "vectorized_docs"]
        )
        logger.warn("Clustering context DataFrame")
        if auto_clustering_enabled == 1:
            logger.warn("Clustering tables with AUTO_CLUSTERING")
            # TODO: embedding model and DBSCAN params in config file
            # TODO: Use an implementation for pyspark
            clustering = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1).fit(
                context_df.select("vectorized_docs").rdd.flatMap(lambda x: x).collect()
            )
            clustering_df = spark.createDataFrame(
                data=np.c_[
                    clustering.labels_.reshape(-1, 1),
                    np.array(
                        context_df.rdd.map(lambda x: x.table_id).collect()
                    ).reshape(-1, 1),
                ].tolist(),
                schema=["table_cluster", "table_id"],
            )

            context_df = context_df.join(clustering_df, "table_id")
        else:
            logger.warn("Clustering tables without AUTO_CLUSTERING")
            context_df = context_df.withColumn("table_cluster", lit(0))

        table_grouping_df = context_df.select(col("table_id"), col("table_cluster"))
        logger.warn("Writing table clustering result to disk.")
        table_grouping_df.write.parquet(output_path, mode="overwrite")
    else:
        logger.warn("Loading table grouping from disk")
        table_grouping_df = spark.read.parquet(output_path)
    return table_grouping_df


def create_table_context(row: Row) -> List:
    """_summary_

    Args:
        row (Row): _description_

    Returns:
        List: _description_
    """
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

    model = api.load("word2vec-google-news-300")
    vectorized_docs = vectorize(tokens, model=model)

    return [
        row.table_id,
        vectorized_docs.tolist(),
    ]
