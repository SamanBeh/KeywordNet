import json


input_path = input("Please enter the file path of keywords FULL info:\n   >>>  ")
line = ""
with open(input_path, "r") as file:
    line = file.readlines()[0]
json_input = json.loads(line.replace("ï»؟", ""))

second_input_path = input("Please enter the file path of keywords info:\n   >>>  ")
line = ""
with open(second_input_path, "r") as file:
    line = file.readlines()[0]

second_json_input = json.loads(line)


output_list = []
for art1 in second_json_input:
    for art2 in json_input:
        if art1.get("keyword") == art2.get("keyword"):
            art1["years"] = art2.get("years")
            output_list.append(art1)

output_file_path = input("Please enter the output file path:\n   >>>  ")
with open(output_file_path, "w") as file:
    file.write(json.dumps(output_list))














#
