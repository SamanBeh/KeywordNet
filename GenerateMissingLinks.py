import pandas as pd
import random

## The missing linkes should be extracted from the previous snapshot of the
## network which you want to predict links for. (if you want to have a negative
## dataset for your predictions)
## for example if you want to have a list of links which were not present
## in 2016 and also is not present in 2017 and the positive dataset is from 2017
## so these links can be in our negative
## dataset

'''
# num_of_edges = int(input("How much edges you want to be generated:  "))
num_of_edges = 100000

old_edges_file_path = input("   Please enter OLD edges file path: \n    >>> ")
new_edges_file_path = input("   Please enter NEW edges file path: \n    >>> ")
# old_nodes_info_file_path = input("   Please enter OLD nodes info file path: \n    >>> ")
# new_nodes_info_file_path = input("   Please enter NEW nodes info file path: \n    >>> ")
output_file_path = input("   Please enter the output file path: \n    >>> ")
'''

def get_missing_links(old_edges_file_path, new_edges_file_path, output_file_path, num_of_edges):
    df_old_edges = pd.read_csv(old_edges_file_path, sep=",")
    df_new_edges = pd.read_csv(new_edges_file_path, sep=",")
    # df_old_nodes = pd.read_csv(old_nodes_info_file_path, sep=",")
    # df_new_nodes = pd.read_csv(new_nodes_info_file_path, sep=",")


    old_nodes = set()
    old_edges = set()
    for s,t in zip(list(df_old_edges.Source), list(df_old_edges.Target)):
        old_edges.add(tuple(sorted((s,t))))
        old_nodes.add(s)
        old_nodes.add(t)

    new_nodes = set()
    new_edges = set()
    for s,t in zip(list(df_new_edges.Source), list(df_new_edges.Target)):
        new_edges.add(tuple(sorted((s,t))))
        new_nodes.add(s)
        new_nodes.add(t)


    nodes = old_nodes & new_nodes
    nodes = list(nodes)
    len_nodes = len(nodes)

    missing_edges = set()
    while True:
        if num_of_edges == 0:
            break
        num_of_edges -= 1
        source = nodes[random.randint(0, len_nodes-1)]
        target = nodes[random.randint(0, len_nodes-1)]
        if source == target:
            num_of_edges += 1
            continue
        edge = tuple(sorted((source, target)))
        if edge not in old_edges and edge not in new_edges and edge not in missing_edges:
            missing_edges.add(edge)
        else:
            num_of_edges += 1
        print("\r{}".format(num_of_edges), end="")


    output_missing_edges = pd.DataFrame(columns=["Source", "Target"])
    cnt = 0
    sources = []
    targets = []
    for edge in list(missing_edges):
        cnt += 1
        print("\r {} added".format(cnt), end="")
        sources.append(edge[0])
        targets.append(edge[1])

    output_missing_edges.Source = sources
    output_missing_edges.Target = targets

    return {"output_dataframe": output_missing_edges, "new_edges": new_edges, "missing_edges": missing_edges}



###################################################################
'''
ot_mss_egs = get_missing_links(old_edges_file_path, new_edges_file_path, output_file_path, num_of_edges)["output_dataframe"]
ot_mss_egs.to_csv(output_file_path, sep=',', index=False)
'''






#
