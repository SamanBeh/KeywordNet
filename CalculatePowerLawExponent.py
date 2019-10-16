import powerlaw
import pandas as pd


# the input file sould contains a squence of nodes' degree
# input_file_path = input("Please enter the file path for the network nodes' degree sequence\n   >>> ").replace("\"", "")
# data = pd.read_csv(input_file_path)
input_dir = input("Enter the root directory for networks info files\n   >>> ").replace("\"", "").replace("\\", "/")


# lines = []
# with open(input_file_path) as file:
#     lines = file.readlines()
#
# lines = [int(line.replace("\n", "")) for line in lines]

# lines = sorted(lines)

results = []
for i in range(18):
    cntr = str(i)
    if i < 10:
        cntr = "0{}".format(cntr)
    data = pd.read_csv("{}/2000_{}-output.csv".format(input_dir, cntr))
    result = powerlaw.Fit(list(data.Degree))
    data = None
    result = "2000_{}-output.csv | {:0.3f}".format(cntr, result.power_law.alpha)
    results.append(result)

print(*results, sep="\n")

# print("Alpha:  {:0.3f}".format(results.power_law.alpha))
# print("Xmin:   {}".format(results.power_law.xmin))
