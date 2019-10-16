import pandas as pd
import sys
import random

def get_edges(df):
    source_ids = list(df["Source"])
    target_ids = list(df["Target"])
    edges = []
    for i in range(len(df)):
        edges.append(tuple(sorted((source_ids[i], target_ids[i]))))

    return edges


step = int(sys.argv[2])
second_file = int(sys.argv[1])
first_file = second_file - step
if second_file < 10:
    second_file = "0{}".format(second_file)
if first_file < 10:
    first_file = "0{}".format(first_file)

first_edges_file = "E:/Research/Dataset/IEEE Dataset/Top-20 IEEE Conferences/2019/conf/2000_{}.csv".format(first_file)
second_edges_file = "E:/Research/Dataset/IEEE Dataset/Top-20 IEEE Conferences/2019/conf/2000_{}.csv".format(second_file)
output_file_path_diff = "E:/Research/Dataset/IEEE Dataset/Top-20 IEEE Conferences/2019/newPredFiles/{}s/diff-{}-{}.txt".format(step, first_file, second_file)
output_file_path_miss = "E:/Research/Dataset/IEEE Dataset/Top-20 IEEE Conferences/2019/newPredFiles/{}s/miss-{}-{}.txt".format(step, first_file, second_file)

print("Loading files {} and {} . . .".format(first_file, second_file))
first_edges = pd.read_csv(first_edges_file)
second_edges = pd.read_csv(second_edges_file)

first_nodes = list(first_edges.Source) + list(first_edges.Target)
first_nodes = set(first_nodes)
second_nodes = list(second_edges.Source) + list(second_edges.Target)
second_nodes = set(second_nodes)

print("Extracting edges . . .")
# first_edges = [tuple(sorted((edge[1]["Source"], edge[1]["Target"]))) for edge in first_edges.iterrows()]
# second_edges = [tuple(sorted((edge[1]["Source"], edge[1]["Target"]))) for edge in second_edges.iterrows()]
first_edges = get_edges(first_edges)
first_edges = set(first_edges)
second_edges = get_edges(second_edges)
second_edges = set(second_edges)
print("Edges Extracted . . .")

print("Finding differences . . . ")
diffs = set()
cntr = 0
lne = len(second_edges)
for edge in second_edges:
    cntr += 1
    print("\r  {}/{}".format(cntr, lne), end="")
    if edge not in first_edges and edge[0] in first_nodes and edge[1] in first_nodes:
        diffs.add(edge)

diffs = ["{},{}\n".format(edge[0], edge[1]) for edge in list(diffs)]

with open(output_file_path_diff, "w") as file:
    file.writelines(diffs)

print("\nGenerating non-edges . . .")
missed_edges = set()
first_nodes = list(first_nodes)
lnd = len(diffs)
cntr = 0
num_of_edges = lnd
while True:
    if num_of_edges == 0:
        break
    cntr += 1
    num_of_edges -= 1
    print("\r  {} tested to create {} edges".format(cntr, lnd), end="")
    source_node = random.choice(first_nodes)
    target_node = random.choice(first_nodes)
    missed_edge = tuple(sorted((source_node, target_node)))
    if source_node != target_node and missed_edge not in second_edges and missed_edge not in missed_edges:
        missed_edges.add(missed_edge)
    else:
        num_of_edges += 1


output_missing_edges = pd.DataFrame(columns=["Source", "Target"])
sources = [edge[0] for edge in list(missed_edges)]
targets = [edge[1] for edge in list(missed_edges)]

output_missing_edges.Source = sources
output_missing_edges.Target = targets

output_missing_edges.to_csv(output_file_path_miss, index=False)

print("\n----------------------\n")
















#
