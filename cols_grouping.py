from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import Row
from pyspark.sql.functions import lit
from typing import List
from openclean.profiling.dataset import dataset_profile

import pandas as pd

import nltk

type_dicts = {'int': 0, 'float': 1, 'str': 2, 'date': 3}

def generate_col_df(row: Row) -> List:
    dirty_df = pd.read_csv(
        row.dirty_path,
        sep=",",
        header="infer",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    column_list = []

    for column_id, column in enumerate(dirty_df.columns.tolist()):
        features = [row.table_id, column_id]

        features = [row.table_id, column_id]
        profiles = dataset_profile(pd.DataFrame(dirty_df[column]))
        features.append(profiles[0]['stats']['totalValueCount'])
        features.append(profiles[0]['stats']['emptyValueCount'])
        features.append(profiles[0]['stats']['distinctValueCount'])
        features.append(float(profiles.stats()['uniqueness'][0]))
        features.append(profiles[0]['stats']['entropy'])

        if len(profiles.types().columns) > 0:
            col_type = profiles.types().columns[0]
            features.append(type_dicts[col_type])
        else:
            features.append(-1)

        for j in range(len(features)):
            if features[j] is None:
                features[j] = -1
        
# ......
        #for value in col_df['col_value'][i]:
        #    for character in list(set(list(str(value)))):
        #        if character not in characters_dictionary:
        #            characters_dictionary[character] = 0.0
        #        characters_dictionary[character] += 1.0
        #    if value not in values_dictionary:
        #        values_dictionary[value] = 0.0
        #    values_dictionary[value] += 1.0



        # (table_id, col_id, totalValueCount, emptyValueCount, distinctValueCount, uniqueness, entropy, type)
        column_list.append(features)

    return column_list

def cluster_cols(col_df: DataFrame, auto_clustering_enabled: int, logger):
    # TODO: dbscan params config
    if auto_clustering_enabled == 1:
        logger.warn("Clustering columns with AUTO_CLUSTERING")
        return col_df
    else:
        logger.warn("Clustering columns without AUTO_CLUSTERING")
        return col_df.withColumn("col_cluster", lit(1))


def col_folding_pyspark(
    csv_paths_df: DataFrame,
    labels_df: DataFrame,
    table_cluster_df: DataFrame,
    column_groups_path: str,
    column_grouping_enabled: int,
    auto_clustering_enabled: int,
) -> DataFrame:
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    nltk.download('stopwords')

    if column_grouping_enabled == 1:
        col_rdd = csv_paths_df.rdd.flatMap(lambda row: generate_col_df(row))
        col_df = col_rdd.toDF(['table_id', 'col_id', 'totalValueCount', 'emptyValueCount', 'distinctValueCount', 'uniqueness', 'entropy', 'type'])

        col_df = cluster_cols(col_df, auto_clustering_enabled, logger)

        logger.warn("Writing column clustering result to disk.")
        col_df.write.parquet(column_groups_path, mode="overwrite")
    else:
        logger.warn("Loading col clustering from disk")
        col_df = spark.read.parquet(column_groups_path)

    return col_df
