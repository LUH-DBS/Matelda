from distutils.dir_util import copy_tree

def move_to_santos(santos_lake_path, santos_query_path, aggregated_sandbox_path):
    copy_tree(aggregated_sandbox_path, santos_lake_path)
    copy_tree(aggregated_sandbox_path, santos_query_path)

santos_lake_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/santos/benchmark/tus_benchmark/datalake"
santos_query_path = "/home/fatemeh/ED-Scale/marshmallow_pipeline/santos/benchmark/tus_benchmark/query"
aggregated_sandbox_path = "/home/fatemeh/ED-Scale/Old_Files/Benchmarks/kaggle_sample_dataset/aggregated_kaggle_lake"

move_to_santos(santos_lake_path, santos_query_path, aggregated_sandbox_path)