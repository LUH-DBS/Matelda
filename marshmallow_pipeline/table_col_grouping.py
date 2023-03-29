from col_grouping_module.col_grouping import group_cols
from cluster_tables import table_grouping

graph_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/mediate_files/graph.gpickle"
lake_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/kaggle/aggregated_kaggle_lake"
separated_lake_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/marshmallow_pipeline/kaggle/separated_kaggle_lake"
table_grouping_dict = table_grouping(graph_path)
col_groups = group_cols(lake_path, table_grouping_dict, separated_lake_path)