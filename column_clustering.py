import csv
from functools import reduce
from itertools import chain
from statistics import median
from typing import Counter, List, Tuple

import nltk
import pandas as pd
from messytables import CSVTableSet, type_guess
from openclean.profiling.dataset import dataset_profile
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import (
    Row,
    StructType,
    StructField,
    IntegerType,
    StringType,
    ArrayType,
)

type_dicts = {
    "Integer": 0,
    "Decimal": 1,
    "String": 2,
    "Date": 3,
    "Bool": 4,
    "Time": 5,
    "Currency": 6,
    "Percentage": 7,
}


def column_clustering_pyspark(
    csv_paths_df: DataFrame,
    table_cluster_df: DataFrame,
    column_groups_path: str,
    column_grouping_enabled: int,
    auto_clustering_enabled: int,
    seed: int,
    save_intermediate_results: bool,
) -> DataFrame:
    """_summary_

    Args:
        csv_paths_df (DataFrame): _description_
        table_cluster_df (DataFrame): _description_
        column_groups_path (str): _description_
        column_grouping_enabled (int): _description_
        auto_clustering_enabled (int): _description_
        seed (int): _description_
        save_intermediate_results (bool): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    nltk.download("stopwords")

    if column_grouping_enabled == 1:
        logger.warn("Creating column features")
        prediction_dfs = []
        column_rdd = csv_paths_df.rdd.flatMap(lambda row: generate_column_df(row))
        column_df = column_rdd.toDF(
            [
                "table_id",
                "column_id",
                "column_type",
                "characters_counter",
                "tokens_counter",
                "median_value_length",
            ]
        )
        column_df = column_df.join(table_cluster_df, "table_id", "inner")

        def group_column_features(key, df: pd.DataFrame):
            """_summary_

            Args:
                key (_type_): _description_
                df (pd.DataFrame): _description_

            Returns:
                pd.DataFrame: _description_
            """
            chars = list(
                set(
                    chain.from_iterable(
                        [list(counter.keys()) for counter in df.characters_counter]
                    )
                )
            )
            tokens = list(
                set(
                    chain.from_iterable(
                        [list(counter.keys()) for counter in df.tokens_counter]
                    )
                )
            )
            return pd.DataFrame(
                {
                    "table_cluster": [key[0]],
                    "characters": [chars],
                    "tokens": [tokens],
                }
            )

        # TODO: Transfering the dictionaries generates a lot of overhead
        grouped_cols_df = (
            column_df.select("table_cluster", "characters_counter", "tokens_counter")
            .groupby("table_cluster")
            .applyInPandas(
                group_column_features,
                schema=StructType(
                    [
                        StructField("table_cluster", IntegerType(), False),
                        StructField("characters", ArrayType(StringType(), True), True),
                        StructField("tokens", ArrayType(StringType(), True), True),
                    ]
                ),
            )
        )

        # TODO: unique is slow
        for c_idx in grouped_cols_df.select("table_cluster").distinct().collect():
            logger.warn("Processing column cluster {}".format(c_idx["table_cluster"]))
            dataset_cluster_column_df = column_df.where(
                column_df.table_cluster == c_idx["table_cluster"]
            )
            dataset_cluster_column_grouped_df = grouped_cols_df.where(
                grouped_cols_df.table_cluster == c_idx["table_cluster"]
            ).collect()

            characters = dataset_cluster_column_grouped_df[0].characters
            tokens = dataset_cluster_column_grouped_df[0].tokens

            logger.warn(
                "Creating feature vectores cluster {}".format(c_idx["table_cluster"])
            )
            dataset_cluster_column_rdd = dataset_cluster_column_df.rdd.map(
                lambda row: create_feature_vector(
                    row,
                    characters,
                    tokens,
                )
            )
            dataset_cluster_column_df = dataset_cluster_column_rdd.toDF(
                ["table_id", "column_id", "table_cluster", "features"]
            )
            logger.warn("Clustering cluster {}".format(c_idx["table_cluster"]))
            column_cluster_prediction_df, num_cluster = cluster_columns(
                dataset_cluster_column_df, auto_clustering_enabled, seed, logger
            )
            prediction_dfs.append(column_cluster_prediction_df)
        # TODO: columns in different table cluster can get clustered in same column cluster number
        column_df = reduce(DataFrame.unionAll, prediction_dfs).select(
            col("table_id"), col("column_id"), col("col_cluster")
        )
        if save_intermediate_results:
            logger.warn("Writing column clustering result to disk.")
            column_df.write.parquet(column_groups_path, mode="overwrite")
    else:
        logger.warn("Loading column clustering from disk")
        column_df = spark.read.parquet(column_groups_path)
        logger.warn("Counting number of columns cluster")
        num_cluster = (
            column_df.agg({"col_cluster": "max"}).collect()[0]["max(col_cluster)"] + 1
        )

    return column_df, num_cluster


def create_feature_vector(row: Row, characters: List[str], tokens: List[str]) -> Row:
    """_summary_

    Args:
        row (Row): _description_
        characters (List[str]): _description_
        tokens (List[str]): _description_

    Returns:
        Row: _description_
    """
    char_list = [Counter(row.characters_counter)[ch] for ch in characters]
    token_list = [Counter(row.tokens_counter)[to] for to in tokens]
    return [
        row.table_id,
        row.column_id,
        row.table_cluster,
        Vectors.dense(
            char_list + token_list + [row.column_type] + [row.median_value_length]
        ),
    ]


def cluster_columns(
    col_df: DataFrame, auto_clustering_enabled: int, seed: int, logger
) -> Tuple[DataFrame, int]:
    """_summary_

    Args:
        col_df (DataFrame): _description_
        auto_clustering_enabled (int): _description_
        seed (int): _description_
        logger (_type_): _description_

    Returns:
        Tuple[DataFrame, int]: _description_
    """
    if auto_clustering_enabled == 1:
        logger.warn("Clustering columns with AUTO_CLUSTERING")
        num_cluster = 10
        kmeans = KMeans(k=num_cluster, seed=0, initMode="k-means||")
        kmeans_model = kmeans.fit(col_df)
        logger.warn("Fitted")
        predictions = kmeans_model.transform(col_df)
        # Removed feature importance. Because performance
        return (
            col_df.join(
                predictions.select("table_id", "column_id", "prediction"),
                ["table_id", "column_id"],
                how="inner",
            ).withColumnRenamed("prediction", "col_cluster"),
            num_cluster,
        )
    else:
        logger.warn("Clustering columns without AUTO_CLUSTERING")
        return col_df.withColumn("col_cluster", lit(0)), 1


def generate_column_df(row: Row) -> List:
    """_summary_

    Args:
        row (Row): _description_

    Returns:
        List: _description_
    """
    dirty_df = pd.read_csv(
        row.dirty_path,
        sep=",",
        header="infer",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        quoting=csv.QUOTE_ALL,
    )

    table_file = open(row.dirty_path, "rb")
    table_set = CSVTableSet(table_file)
    row_set = table_set.tables[0]
    types = type_guess(row_set.sample)

    column_list = []

    for column_id, column in enumerate(dirty_df.columns.tolist()):
        indices = [row.table_id, column_id]
        characters_counter = Counter()
        tokens_counter = Counter()
        features = []

        features.append(get_column_type(types, column_id))

        value_length_sum = []
        column_length = 0
        for value in dirty_df[column].values:
            column_length += 1
            char_list = list(set(list(str(value))))
            if " " in char_list:
                char_list.remove(" ")

            characters_counter.update(char_list)

            value_length_sum.append(len(str(value)))

            tokens = str(value).split()
            tokens = list(
                filter(lambda token: len(token) > 1, tokens)
            )  # Makes sure no chars are inside the token list
            if " " in tokens:
                tokens.remove(" ")

            tokens_counter.update(tokens)

        for key in characters_counter:
            characters_counter[key] /= column_length
        for key in tokens_counter:
            tokens_counter[key] /= column_length

        features.append(dict(characters_counter))
        features.append(dict(tokens_counter))
        features.append(median(value_length_sum))

        column_list.append(indices + features)

    return column_list


def get_column_type(types: CSVTableSet, column_id: int) -> int:
    """_summary_

    Args:
        types (CSVTableSet): _description_
        column_id (int): _description_

    Returns:
        int: _description_
    """
    if types[column_id]:
        col_type = str(types[column_id])
        if "Date" in col_type:
            return type_dicts["Date"]
        elif "Time" in col_type:
            return type_dicts["Type"]
        else:
            return type_dicts[col_type]
    else:
        return -1
