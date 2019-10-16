## this program gets a sequence of nodes ids and find their degree in the other given dataframe
import pandas as pd


new_edges_file_path = input("Enter the newly joined edges file path:\n   >>> ")
nodes_info_file_path = input("Enter nodes info file path:\n   >>> ")


new_edges = pd.read_csv(new_edges_file_path)
nodes_info = pd.read_csv(nodes_info_file_path)

nodes = set(list(new_edges.Source) + list(new_edges.Target))

nodes_degrees = []

for i in range(len(nodes_info)):
    if nodes_info.iloc[i]["NodeId"] in nodes:
        nodes_degrees.append((nodes_info.iloc[i]["NodeId"], nodes_info.iloc[i]["Degree"]))

nodes_degrees = sorted(nodes_degrees, key=lambda e:e[1])
print(*nodes_degrees, sep="\n")
print("Number of nodes which has new connections: {}".format(len(nodes_degrees)))

output_df = pd.DataFrame(columns=["NodeId", "Degree"])
for nd in nodes_degrees:
    output_df = output_df.append({"NodeId": nd[0], "Degree": nd[1]}, ignore_index=True)

print(output_df)

output_df.to_csv("D:/NewNodes-Degrees.txt", index=False)










#
