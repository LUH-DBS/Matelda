from typing import List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import Row


def generate_table_ground_truth(row: Row) -> List[Tuple[int, int, int, bool]]:
    """_summary_

    Args:
        row (Row): _description_

    Returns:
        List[Tuple[int, int, int, bool]]: _description_
    """
    tables_ground_truth = []
    dirty_df = pd.read_csv(
        row.dirty_path,
        sep=",",
        header="infer",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    clean_df = pd.read_csv(
        row.clean_path,
        sep=",",
        header="infer",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    labels_df = dirty_df.where(dirty_df.values == clean_df.values).notna() * 1

    for rowIndex, row2 in labels_df.iterrows():  # iterate over rows
        for columnIndex, value in row2.items():
            tables_ground_truth.append(
                (row.table_id, labels_df.columns.get_loc(columnIndex), rowIndex, value)
            )
            # pyspark cant safe a list in a Dataframe (type infering problem)
    return tables_ground_truth


def generate_labels_pyspark(
    csv_paths_df: DataFrame, labels_path: str, extract_labels_enabled: bool
) -> DataFrame:
    """_summary_

    Args:
        csv_paths_df (DataFrame): _description_
        labels_path (str): _description_
        extract_labels_enabled (bool): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    if extract_labels_enabled:
        logger.warn("Extracting labels")
        labels_rdd = csv_paths_df.rdd.flatMap(
            lambda row: generate_table_ground_truth(row)
        )
        labels_df = labels_rdd.toDF(["table_id", "column_id", "row_id", "ground_truth"])
        logger.warn("Writing labels to file")
        labels_df.write.parquet(labels_path, mode="overwrite")
    else:
        logger.warn("Loading labels from disk")
        labels_df = spark.read.parquet(labels_path)

    return labels_df