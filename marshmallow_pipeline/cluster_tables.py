import os
import pickle
import networkx as nx
import networkx.algorithms.community as nx_comm


def table_grouping(graph_path):
    print("table_grouping")
    print("community detection")
    handle = open(graph_path, 'rb')
    G = pickle.load(handle)
    # G = nx.read_gpickle(graph_path)
    try:
        comp = nx_comm.louvain_communities(G)
        table_group_dict = {}
        table_group_dict_key = 0
        # Print the partitions
        print("creating table_group_dict")
        for community in comp:
            table_group_dict[table_group_dict_key] = []
            for table in community:
                table_group_dict[table_group_dict_key].append(table)
            table_group_dict_key += 1
    except:
        print("community detection failed")
        table_group_dict = {}
        table_group_dict_key = 0
        table_group_dict[table_group_dict_key] = []
        for table in G.nodes():
            table_group_dict[table_group_dict_key].append(table)

    with open(os.path.join(os.path.dirname(graph_path), 'table_group_dict.pickle'), 'wb') as handle:
        pickle.dump(table_group_dict, handle)
    return table_group_dict
