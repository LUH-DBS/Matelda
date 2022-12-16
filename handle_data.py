import os

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import monotonically_increasing_id


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
                        csv_paths.append((dirty_path, clean_path, table, parent))
    csv_paths_df = (
        spark.createDataFrame(
            data=csv_paths,
            schema=["dirty_path", "clean_path", "table_name", "parent"],
        )
        .sort("table_name")
        .withColumn("table_id", monotonically_increasing_id())
        .select("table_id", "dirty_path", "clean_path", "table_name", "parent")
        .repartition(spark.sparkContext.defaultParallelism)
    )
    print(csv_paths_df.count())
    print(csv_paths_df.select('table_name').distinct().count())
    return csv_paths_df
