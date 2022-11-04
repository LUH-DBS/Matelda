import json
import os
import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.mllib.evaluation import MulticlassMetrics


def evaluate_pyspark(
    y_predictions_df: DataFrame, labels_df: DataFrame, result_path: str
) -> None:
    """_summary_

    Args:
        y_predictions_df (DataFrame): _description_
        labels_df (DataFrame): _description_
        result_path (str): _description_
    """
    spark = SparkSession.getActiveSession()
    log4jLogger = spark._jvm.org.apache.log4j
    logger = log4jLogger.LogManager.getLogger(__name__)

    metrics = MulticlassMetrics(
        y_predictions_df.join(labels_df, ["table_id", "column_id", "row_id"])
        .select(
            col("ground_truth").cast(FloatType()).alias("ground_truth"),
            col("prediction").cast(FloatType()).alias("prediction"),
        )
        .rdd
    )
    evaluation_dict = {}
    evaluation_dict["Accuracy"] = metrics.accuracy
    evaluation_dict["Precision (True)"] = metrics.precision(1.0)
    evaluation_dict["Recall (True)"] = metrics.recall(1.0)
    evaluation_dict["F-Measure (True)"] = metrics.fMeasure(1.0)
    evaluation_dict["True positive rate (True)"] = metrics.truePositiveRate(1.0)
    evaluation_dict["False positive rate (True)"] = metrics.falsePositiveRate(1.0)
    evaluation_dict["Precision (False)"] = metrics.precision(0.0)
    evaluation_dict["Recall (False)"] = metrics.recall(0.0)
    evaluation_dict["F-Measure (False)"] = metrics.fMeasure(0.0)
    evaluation_dict["True positive rate (False)"] = metrics.truePositiveRate(0.0)
    evaluation_dict["False positive rate (False)"] = metrics.falsePositiveRate(0.0)
    evaluation_dict["Confusion matrix"] = metrics.confusionMatrix().toArray().tolist()

    for result in evaluation_dict.items():
        logger.warn("{}: {}".format(result[0], result[1]))
    logger.warn("Saving evaluation results to disk")

    with open(os.path.join(result_path, "evaluation_results.json"), "w") as outfile:
        outfile.write(json.dumps(evaluation_dict))
