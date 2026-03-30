import json

import numpy as np
import pandas as pd

with open("feature_weights.json") as f:
    feature_weights = json.load(f)
with open("train.rubric.features_hard.aggregate.json") as f:
    data_original = json.load(f)["all"]
with open("oecd.rubric.features.aggregate.json") as f:
    data_oecs = json.load(f)["all"]
with open("subsample.features.aggregate.json") as f:
    data_subsample = json.load(f)["all"]

sum_w = sum(feature_weights.values())

tm = "_mean"


def print_similarity_info(data1, data2):
    total = 0
    for k, v in feature_weights.items():
        dist = np.abs(data1[k + tm] - data2[k + tm])
        total += float(dist) / sum_w
        print(k, f"{float(dist):0.3f}")
    print("total_dist", f"{float(total):0.3f}")


print("---ORIGINAL---")
print_similarity_info(data_original, data_oecs)
print("---SUBSAMPLE---")
print_similarity_info(data_subsample, data_oecs)


def make_latex_table(data1, data2, data3):
    vs1 = []
    vs2 = []
    vs3 = []
    dif1 = []
    dif2 = []
    ks = list(feature_weights.keys())
    for k in ks:
        vs1.append(data1[k + tm])
        vs2.append(data2[k + tm])
        vs3.append(data3[k + tm])
        dif1.append(float(np.abs(data1[k + tm] - data3[k + tm])))
        dif2.append(float(np.abs(data2[k + tm] - data3[k + tm])))
    data_all = [{} for x in vs1]

    for data, name in (
    zip([dif1, dif2], ["abs difference original vs confidential", "abs difference selected vs confidential"])):
        for d, dic in zip(data, data_all):
            dic[name] = d

    st = pd.DataFrame().from_records(data_all, index=[x.replace("_", " ").replace("input/", "") for x in ks]).to_latex(
        float_format="%.03f")
    with open("table.txt", "w") as f:
        f.write(st)


make_latex_table(data_original, data_subsample, data_oecs)
