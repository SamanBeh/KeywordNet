import pandas as pd
import os


# Read 'N'Jcorrdiff.csv file to add community info to it
edges_file_path = input(" Please enter the edges info file path: [e.g. 5Jcorrdiff.csv]\n    >>> ").replace("\\", "/")
edges_info = pd.read_csv(edges_file_path)

# Read all 'N' steps backward to extarct community info
# i.e. if you want to have comm info to predict 2017 you
# should use comm info of 2016 and before
comm_info_files_paths = list(os.walk(input(" Please enter the root path for community info files: \n    >>> ")))[0]
steps = int(input(" Please enter the number of steps you want to check backward:  ")) + 1

root_path = comm_info_files_paths[0]
comm_info_files_paths = ["{}\\{}".format(root_path, fp) for fp in comm_info_files_paths[2]]
comm_info_files_paths = list(reversed(comm_info_files_paths))[1:steps] # Mind the [1:] | I used it to remove 2017 from list
# print(*comm_info_files_paths, sep="\n")

dfs_list = []
for path in comm_info_files_paths:
    df = pd.read_csv(path)
    dct = {"nodes": set(df["id"]), "df": df}
    dfs_list.append(dct)

# check all edges in 'N'Jcorrdiffto to see if their
# source and target nodes were in the same community or not

sources = list(edges_info["Source"])
targets = list(edges_info["Target"])

final_df = pd.DataFrame()

len_srcs = len(sources)
stat_cntr = 0
for sr,tg in zip(sources, targets):
    stat_cntr += 1
    print("\r{}/{}".format(stat_cntr, len_srcs), end="")
    comm_score = 0
    cntr = len(dfs_list)
    for year in dfs_list:
        if sr in year["nodes"] and tg in year["nodes"]:
            df = year["df"]
            sr_comm = int(df[df["id"] == sr]["modularity_class"])
            tg_comm = int(df[df["id"] == tg]["modularity_class"])
            if sr_comm == tg_comm:
                comm_score += cntr
            cntr -= 1

    final_df = final_df.append({"Source": sr, "Target": tg, "comm_score": comm_score}, ignore_index=True)

print()
edges_info["comm_gephi"] = 0
len_finaldf = len(final_df)
stat_cntr = 0
for row in final_df.iterrows():
    stat_cntr += 1
    print("\r{}/{}".format(stat_cntr, len_finaldf), end="")
    src = int(row[1]["Source"])
    tgt = int(row[1]["Target"])
    st = sorted([src,tgt]) # We know that the Source|Target of the edges_info file are sorted
    edges_info.loc[(edges_info["Source"] == st[0]) | (edges_info["Target"] == st[1]), ["comm_gephi"]] = int(row[1]["comm_score"])



ext = edges_file_path[edges_file_path.rfind("."):]
output_file_path = edges_file_path[:edges_file_path.rfind(".")] + "-WithComm" + ext

print("\nWriting results to file . . .")
edges_info.to_csv(output_file_path, index=False)














#
