from typing import Counter, List

import nltk
import operator
import pandas as pd
from collections import Counter
from functools import reduce
from openclean.profiling.dataset import dataset_profile
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.types import Row

type_dicts = {"int": 0, "float": 1, "str": 2, "date": 3}


def generate_column_df(row: Row) -> List:
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
        characters_dictionary = {}
        values_dictionary = {}

        features = [row.table_id, column_id]
        profiles = dataset_profile(pd.DataFrame(dirty_df[column]))
        features.append(profiles[0]["stats"]["totalValueCount"])
        features.append(profiles[0]["stats"]["emptyValueCount"])
        features.append(profiles[0]["stats"]["distinctValueCount"])
        features.append(float(profiles.stats()["uniqueness"][0]))
        features.append(float(profiles[0]["stats"]["entropy"]))

        if len(profiles.types().columns) > 0:
            col_type = profiles.types().columns[0]
            features.append(type_dicts[col_type])
        else:
            features.append(-1)

        for j in range(len(features)):
            if features[j] is None:
                features[j] = -1

        for value in dirty_df[column].values:
           for character in list(set(list(str(value)))):
               if character not in characters_dictionary:
                   characters_dictionary[character] = 0.0
               characters_dictionary[character] += 1.0
           if value not in values_dictionary:
               values_dictionary[value] = 0.0
           values_dictionary[value] += 1.0

        features.append(characters_dictionary)
        features.append(list(characters_dictionary.keys()))
        features.append(values_dictionary)
        features.append(list(values_dictionary.keys()))

        # (table_id, col_id, table_cluster, totalValueCount, emptyValueCount, distinctValueCount, uniqueness, entropy, type, char_dict, val_dict)
        column_list.append(features)

    return column_list


def cluster_columns(col_df: DataFrame, auto_clustering_enabled: int, logger):
    # TODO: dbscan params config
    col_df.show()
    if auto_clustering_enabled == 1:
        # TODO: Add DBSCAN
        logger.warn("Clustering columns with AUTO_CLUSTERING")
        return col_df.withColumn("col_cluster", lit(1))
    else:
        logger.warn("Clustering columns without AUTO_CLUSTERING")
        return col_df.withColumn("col_cluster", lit(1))


def column_clustering_pyspark(
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

    nltk.download("stopwords")

    if column_grouping_enabled == 1:
        logger.warn("Creating column features")
        column_rdd = csv_paths_df.rdd.flatMap(lambda row: generate_column_df(row))
        column_df = column_rdd.toDF(
            [
                "table_id",
                "col_id",
                "totalValueCount",
                "emptyValueCount",
                "distinctValueCount",
                "uniqueness",
                "entropy",
                "type",
                "char_dict",
                "char_dict_keys",
                "val_dict",
                "val_dict_keys",
            ]
        )
        column_df = column_df.join(table_cluster_df, 'table_id', 'inner').show()        
        logger.warn("Building char and val dict")
        # TODO: vectorize
        #column_df = column_df.withColumns({'char_dict_keys': None, 'val_dict_keys': None})
        #char_dict_keys= column_df.select('char_dict_keys').rdd.reduce(lambda x,y: list(set(x) | set(y)))
        #print(char_dict_keys)
        #val_dict_keys = dict(reduce(operator.add, map(Counter, column_df.select('val_dict').collect())))
        #print(char_dict_keys)
        #column_df = column_df.withColumns({'col_profile_char': None, 'col_profile_val': None})

        column_df.drop('char_dict', 'char_dict_keys', 'val_dict', 'val_dict_keys')

        column_df = cluster_columns(column_df, auto_clustering_enabled, logger)

        logger.warn("Writing column clustering result to disk.")
        column_df.write.parquet(column_groups_path, mode="overwrite")
    else:
        logger.warn("Loading column clustering from disk")
        column_df = spark.read.parquet(column_groups_path)

    return column_df
