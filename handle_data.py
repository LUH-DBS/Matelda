import os

from pyspark.sql import DataFrame, SparkSession


def generate_csv_paths(sandbox_path: str) -> DataFrame:
    """_summary_

    Args:
        sandbox_path (str): _description_

    Returns:
        DataFrame: _description_
    """
    spark = SparkSession.getActiveSession()
    log4j_logger = spark._jvm.org.apache.log4j
    logger = log4j_logger.LogManager.getLogger(__name__)

    table_id = 0
    csv_paths = []
    sandbox_children_path = [
        (os.path.join(sandbox_path, dir), dir) for dir in os.listdir(sandbox_path)
    ]
    for child_path, parent in sandbox_children_path:
        if os.path.isdir(child_path) and not child_path.startswith("."):
            dirty_path = child_path + "/dirty_clean.csv"
            clean_path = child_path + "/clean.csv"
            if (
                os.path.exists(dirty_path)
                and os.path.exists(clean_path)
                and os.path.isfile(dirty_path)
                and os.path.isfile(clean_path)
            ):
                csv_paths.append((table_id, dirty_path, clean_path, parent, parent))
                table_id += 1

    csv_paths_df = spark.createDataFrame(
        data=csv_paths,
        schema=["table_id", "dirty_path", "clean_path", "table_name", "parent"],
    )

    logger.warn(
        f"Partitions csv_paths_df: {csv_paths_df.rdd.getNumPartitions()}"
    )
    return csv_paths_df
