from scourgify.col_grouping_module.col_grouping import group_cols
from scourgify.table_grouping_module.cluster_tables import table_grouping

graph_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/graph.gpickle"
lake_path = "/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/scourgify/datalake"
table_grouping_dict = table_grouping(graph_path)
group_cols(lake_path, table_grouping_dict)