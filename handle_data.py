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
    csv_paths = []
    sandbox_children_path = [
        (os.path.join(sandbox_path, dir), dir) for dir in os.listdir(sandbox_path)
    ]
    table_id = 0
    for child_path, parent in sandbox_children_path:
        if os.path.isdir(child_path) and not child_path.startswith("."):
            table_dirs = [
                (os.path.join(child_path, dir), dir) for dir in os.listdir(child_path)
            ]
            for table_path, table in table_dirs:
                if os.path.isdir(table_path) and not table_path.startswith("."):
                    dirty_path = table_path + "/dirty.csv"
                    clean_path = table_path + "/" + table + ".csv"
                    if (
                        os.path.exists(dirty_path)
                        and os.path.exists(clean_path)
                        and os.path.isfile(dirty_path)
                        and os.path.isfile(clean_path)
                    ):
                        csv_paths.append(
                            (table_id, dirty_path, clean_path, table, parent)
                        )
                        table_id += 1
    csv_paths_df = spark.createDataFrame(
        data=csv_paths,
        schema=["table_id", "dirty_path", "clean_path", "table_name", "parent"],
    )
    return csv_paths_df