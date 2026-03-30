import json
from itertools import chain

import numpy as np


def dist_f(x, y, weight, std):
    x /= std
    x /= np.sum(x)
    y /= std
    y /= std.sum(y)
    return weight * np.square(x - y)


def match_neighbors(data, data_ref, feature_weights, stds):
    ks = list(feature_weights.keys())
    feature_weights = np.array([feature_weights[k] for k in ks])
    stds = np.array([stds[k] for k in ks])
    for label in set(x["label"] for x in data):
        data_subs = [item for item in data if item["label"] == label]
        features = np.array([[item[k] for k in ks] for item in data_subs])
        features_ref = np.array([[item[k] for k in ks] for item in data_ref if item["label"] == label])
        np.random.seed(42)
        np.random.shuffle(features)
        np.random.shuffle(features_ref)
        for i in range(len(features)):
            dp = data_subs[i]
            dpf = features[i]
            dists_own = dist_f(features, dpf, feature_weights, stds)[:, 0]
            dists_ref = dist_f(features_ref, dpf, feature_weights, stds)[:, 0]
            is_ref = np.concatenate([np.zeros(dists_own.shape), np.ones(dists_ref.shape)])
            vs_inds = np.argsort(np.concatenate([dists_own, dists_ref]))
            dp["score"] = -float(np.mean(np.nonzero(is_ref[vs_inds])[0][:5]))


def match_clusters(data_train, features_oecd, feature_weights, stds):
    for i, x in enumerate(data_train):
        tm = "_mean"
        clusters_to_match = features_oecd[x["label"]]
        dists = [
            sum(float(dist_f(x[k], features_to_match[k + tm], v, stds[k])) / sum_w for k, v in feature_weights.items())
            for features_to_match in clusters_to_match]
        matched_cluster = int(np.argmin(dists))
        x["matched_cluster"] = matched_cluster
        score = dists[matched_cluster]
        x["score"] = -score


with open("./complete_generated_data.json", "r", encoding="utf-8") as f:
    # with open("../final_trainset/train.rubric0.en.json", "r", encoding="utf-8") as f:
    data_original = json.load(f)
for i, x in enumerate(data_original):
    x["id"] = i
inds = np.arange(len(data_original))
# print(np.unique([x["fileid"] for x in data_original], return_counts=True))
np.random.seed(42)
np.random.shuffle(inds)
cs = np.unique([x["label"] for x in data_original], return_counts=True)[1]
minc = np.min(cs)
inds_selected = []
inds_selected += [i for i in inds if data_original[i]["label"] == 0][:minc]
inds_selected += [i for i in inds if data_original[i]["label"] == 1][:minc]
inds_selected += [i for i in inds if data_original[i]["label"] == 2][:minc]
inds_selected = sorted(inds_selected)
data_original = [data_original[i] for i in inds_selected]

with open("complete_generated_data.features.json", encoding="utf-8") as f:
    # with open("train.en.filtered.rubric.features.json", encoding="utf-8") as f:
    data_train = json.load(f)
    data_train = [data_train[i] for i in inds_selected]

for i, x in enumerate(data_train):
    x["ind"] = i
    x["id"] = data_original[i]["id"]
    x["fileid"] = data_original[i]["fileid"]
unique_labels = np.unique([x["label"] for x in data_train]).tolist()
unique_domains = np.unique([x["fileid"].split("/")[1] for x in data_train]).tolist()

with open("train_non_subsample.rubric.json", "r") as f:
    data = json.load(f)
    print(len(data), len(data_train))
with open("train_subsample.rubric.json", "r") as f:
    data = json.load(f)
    print(len(data), len(data_train))

subsample = []
for label in unique_labels:
    for domain in unique_domains:
        subs = [x for x in data_train if x["label"] == label and x["fileid"].split("/")[1] == domain]
        subsample += subs[:len(subs) // 3]
with open("non_subsample.json", "w") as f:
    json.dump(subsample, f, ensure_ascii=False, indent=2)
with open("train_non_subsample.rubric.json", "w") as f:
    json.dump([data_original[x["ind"]] for x in subsample], f, ensure_ascii=False, indent=2)

for x in data_train:
    if "recall_2gram_input/FC_input/answer" in x:
        x["recall_2gram_input/rubrics/FC_input/answer"] = x.pop("recall_2gram_input/FC_input/answer")
    if "recall_2gram_input/NC_input/answer" in x:
        x["recall_2gram_input/rubrics/NC_input/answer"] = x.pop("recall_2gram_input/NC_input/answer")

with open("feature_weights.json") as f:
    feature_weights = json.load(f)

with open(
        "/home/bojar/diplomka/granty/openeurollm/sensemaking-2026/oecd-pisa-test-for-sensemaking-2026.CONFIDENTIAL/pisa_testset.rubric.json.features.json",
        encoding="utf-8") as f:
    features_oecd = json.load(f)

# means = {k: np.mean([x[k] for x in data_train]) for k in feature_weights.keys()}
stds = {k: np.std([x[k] for x in data_train]) for k in feature_weights.keys()}

sum_w = sum(feature_weights.values())
if False:
    features_oecd = {0: features_oecd["0"], 1: features_oecd["1"], 2: features_oecd["2"]}
    match_clusters(data_train, data_original, features_oecd, feature_weights, stds)
else:
    # features_oecd = [x for x in features_oecd if x["lang"] == "en"]
    match_neighbors(data_train, data_original, features_oecd, feature_weights, stds)
print(len(data_train), len(features_oecd))
data_train = sorted(data_train, key=lambda x: x["score"], reverse=True)
# print([x["ind"] for x in data_train if x["label"] == 2])
r = {"2": [x for x in data_train if x["label"] == 2], "1": [x for x in data_train if x["label"] == 1],
     "0": [x for x in data_train if x["label"] == 0]}

with open("data_relevance.json", "w") as f:
    json.dump(r, f, ensure_ascii=False, indent=2)
print(unique_labels, unique_domains)

subsample = []
for label in unique_labels:
    for domain in unique_domains:
        subs = [x for x in data_train if x["label"] == label and x["fileid"].split("/")[1] == domain]
        subsample += subs[:len(subs) // 3]
print("score", np.mean([x["score"] for x in subsample]), np.mean([x["score"] for x in data_train]))
with open("subsample.json", "w") as f:
    json.dump(subsample, f, ensure_ascii=False, indent=2)
with open("train_subsample.rubric.json", "w") as f:
    json.dump([data_original[x["ind"]] for x in subsample], f, ensure_ascii=False, indent=2)

print(len(subsample), len(data_train))
sep = "\nSEPARATOR|SEPARATOR\n"
matchf = lambda x: x["input"]["context"] + sep + x["input"]["question"]
splitp = lambda v: int(len(v) * 0.2)
splitr = {k: [[x for x in v1 if x["fileid"].split("/")[1] == domain] for domain in unique_domains] for k, v1 in
          r.items()}

fewshot_set_not_selected = {k: list(chain(*[[data_original[x["ind"]] for x in v1[splitp(v1):]] for v1 in v])) for k, v
                            in splitr.items()}
fewshot_dataset_questions_fewshot_set_not_selected = {
    k: list(chain(*[[matchf(data_original[x["ind"]]) for x in v1[splitp(v1):]] for v1 in v])) for k, v in
    splitr.items()}
fewshot_set_selected = {k: list(chain(*[[data_original[x["ind"]] for x in v1[:splitp(v1)]] for v1 in v])) for k, v in
                        splitr.items()}
fewshot_dataset_questions_selected = {
    k: list(chain(*[[matchf(data_original[x["ind"]]) for x in v1[:splitp(v1)]] for v1 in v])) for k, v in
    splitr.items()}


def get_dataset(name, fewshot_dataset_questions, fewshot_set, number=6, outf=None):
    appearing_in_all = sorted(set.intersection(*[set(x) for x in fewshot_dataset_questions.values()]))
    print("appearing_in_all", len(appearing_in_all))
    appearing_in_all = [appearing_in_all[i] for i in
                        np.unique([x.split(sep)[0] for x in appearing_in_all], return_index=True)[1]]
    fewshot_dataset = []
    np.random.seed(42)
    np.random.shuffle(appearing_in_all)
    appearing_in_all_unique_domain = np.unique(
        [[[y for y in x if matchf(y) == contextquestion][0] for x in fewshot_set.values()][0]["fileid"].split("/")[1]
         for contextquestion in appearing_in_all], return_inverse=True)
    appearing_in_all_inds = []
    for i in range(len(appearing_in_all_unique_domain[0])):
        appearing_in_all_inds.extend(
            [j for j, x in enumerate(appearing_in_all_unique_domain[1]) if x == i][:number // 3])
    appearing_in_all = [appearing_in_all[x] for x in appearing_in_all_inds]
    for contextquestion in appearing_in_all[:number]:
        examples = list(chain(*[[y for y in x if matchf(y) == contextquestion][:1] for x in fewshot_set.values()]))
        examples = dict(examples[0], answers=[(x["input"]["answer"], x["label"]) for x in examples])
        fewshot_dataset.append(examples)

    with open(f"fewshot_dataset_{name}.json" if outf is None else outf, "w") as f:
        json.dump(fewshot_dataset, f, indent=2, ensure_ascii=False)
    return fewshot_dataset


fewshot_datasets = get_dataset("selected", fewshot_dataset_questions_selected, fewshot_set_selected)
print(len(fewshot_set_selected))
fewshot_datasetns = get_dataset("not_selected", fewshot_dataset_questions_fewshot_set_not_selected,
                                fewshot_set_not_selected)
assert len(set.intersection(set([x["input"]["context"] for x in fewshot_datasets]),
                            set([x["input"]["context"] for x in fewshot_datasetns]))) == 0
