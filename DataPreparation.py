##########################
## EXTRACTING  KEYWORDS ##
##########################
import json
import os
import pandas as pd

def get_first_line_of_file(input_file_path):
    with open(input_file_path, "r") as file:
        return file.readlines()[0]

input_root_path = input("please input the root path for json files: [e.g.: E:/jsonfiles/]:\n   >>>  ").strip()

files = list(os.walk(input_root_path))[0]

json_files = []
print(files[0])
parent_dir = files[0] + "\\"

for file_name in files[2]:
    json_files.append(json.loads(get_first_line_of_file(parent_dir + file_name)))


########################################################################
######   Extracting keywords and their first year of apearance   #######
########################################################################
conf_or_jour = input("Conference or Journal?  [C/J]:  ")

inspec_controlled_indexing_keywords = set()
kw_year_dict = dict() # {"keyword": {"year": 2017, "weight": 156}, ....}

lenjf = len(json_files)
cntr = 1
cnt = 0
for jf in json_files:
    print("\r{}/{}".format(cntr,lenjf), end="")
    cntr += 1
    for article in jf:
        try:
            for kw in article.get("inspec_controlled_indexing"):
                if conf_or_jour.strip().lower() in ["j", "jour", "journal"]:
                    if kw_year_dict.get(kw.strip().lower()) != None:
                        kw_year_dict[kw.strip().lower()]["weight"] += 1
                        if int(article["year"]) < kw_year_dict[kw.strip().lower()]["year"]:
                            kw_year_dict[kw.strip().lower()]["year"] = int(article["year"])
                    else:
                        kw_year_dict[kw.strip().lower()] = dict()
                        kw_year_dict[kw.strip().lower()]["year"] = int(article["year"])
                        kw_year_dict[kw.strip().lower()]["weight"] = 1
                else:
                    if kw_year_dict.get(kw.strip().lower()) != None:
                        kw_year_dict[kw.strip().lower()]["weight"] += 1
                        if int(article["conf_year"]) < kw_year_dict[kw.strip().lower()]["year"]:
                            kw_year_dict[kw.strip().lower()]["year"] = int(article["conf_year"])
                    else:
                        kw_year_dict[kw.strip().lower()] = dict()
                        kw_year_dict[kw.strip().lower()]["year"] = int(article["conf_year"])
                        kw_year_dict[kw.strip().lower()]["weight"] = 1
                inspec_controlled_indexing_keywords.add(kw.strip().lower())

        except Exception as e:
            cnt+=1
            print(cnt)
            # raise


print("\n------------------\n")
print("Num of Keywords: {}".format(len(inspec_controlled_indexing_keywords)))
# print(list(inspec_controlled_indexing_keywords)[:10])

# print(kw_year_dict)


df = pd.DataFrame(columns=["keyword", "birth_year", "weight", "newid"])

df.keyword = list(kw_year_dict.keys())
lst1 = []
lst2 = []
for key in list(kw_year_dict.keys()):
    lst1.append(kw_year_dict[key].get("year"))
    lst2.append(kw_year_dict[key].get("weight"))
df.birth_year = lst1
df.weight = lst2

# print(df.iloc[1300:1400,:])
# print(list(df.keyword)[:5])
# print(df.head())


df = df.sort_values(["birth_year", "weight"], ascending=[True, False])
df = df.reset_index(drop=True)
df.newid = [1000000 + i for i in reversed(range(len(df)))]
# print(df.iloc[1300:1400,:])
# print(list(df.keyword)[:5])
print(df.head())
print("\n   .....   \n")
print(df.tail())



otpt = input("\nPlease enter the output file path:\n   >>>  ").strip()
if len(otpt) > 0:
    df.to_csv(otpt, sep=",", index=False, encoding='utf-8-sig')



###############################
####  Export Keywords-Ids  ####
###############################

import json
import codecs
kwd_id_dict_list = []
for i in range(len(df)):
    kwd_id_dict_list.append({"id": int(df.iloc[i].newid), "keyword": df.iloc[i].keyword})

with codecs.open(input("Please enter the output file path for (keyword, id):\n   >>>  "), "w", encoding="utf8") as file:
    file.write(json.dumps(kwd_id_dict_list))












#
