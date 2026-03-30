import argparse
import json
import os
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score
)


def evaluateE3(fol_subsample, fol_non_subsample, alpha=0.05):
    accuracies_subsample_testl = []
    for x in os.listdir(fol_subsample):
        with open(os.path.join(fol_subsample, x), "r") as f:
            accuracies_subsample_testl.append(np.array(json.load(f)))
    accuracies_non_subsample_testl = []
    for x in os.listdir(fol_non_subsample):
        with open(os.path.join(fol_non_subsample, x), "r") as f:
            accuracies_non_subsample_testl.append(np.array(json.load(f)))
    minl = min([len(x) for x in accuracies_non_subsample_testl + accuracies_subsample_testl])
    for i in range(len(accuracies_subsample_testl)):
        accuracies_subsample_testl[i] = accuracies_subsample_testl[i][:minl]
    for i in range(len(accuracies_non_subsample_testl)):
        accuracies_non_subsample_testl[i] = accuracies_non_subsample_testl[i][:minl]
    plt.plot(np.subtract(accuracies_subsample_testl[0], accuracies_non_subsample_testl[0])[1:])
    plt.savefig("test_accuracies.png")
    plt.figure()
    t_stat, p_value = stats.wilcoxon(np.concatenate(accuracies_subsample_testl[:len(accuracies_non_subsample_testl)]),
                                     np.concatenate(accuracies_non_subsample_testl[:len(accuracies_subsample_testl)]),
                                     alternative='greater')
    print(p_value, p_value < alpha,
          np.mean(np.subtract(accuracies_subsample_testl[0], accuracies_non_subsample_testl[0])),
          np.mean(np.subtract(accuracies_subsample_testl[1], accuracies_non_subsample_testl[1])),
          np.mean(np.subtract(np.concatenate(accuracies_subsample_testl[:len(accuracies_non_subsample_testl)]),
                              np.concatenate(accuracies_non_subsample_testl[:len(accuracies_subsample_testl)]))))
    np.random.seed(43)
    vss = []
    vs_n = []
    order = [(np.random.permutation(len(accuracies_subsample_testl)),
              np.random.permutation(len(accuracies_non_subsample_testl))) for i in range(minl)]
    minl2 = min(len(accuracies_subsample_testl), len(accuracies_non_subsample_testl))
    print(minl, minl2)
    for indi in range(minl2):
        for i in range(minl):
            inds = order[i][0][indi], order[i][1][indi]
            vss.append(accuracies_subsample_testl[inds[0]][i])
            vs_n.append(accuracies_non_subsample_testl[inds[1]][i])

    a1 = np.max(accuracies_subsample_testl, 0) - np.min(accuracies_non_subsample_testl, 0)
    a2 = np.min(accuracies_subsample_testl, 0) - np.max(accuracies_non_subsample_testl, 0)
    plt.figure()
    plt.plot(a1)
    plt.plot(a2)
    plt.savefig("test_accuracies_best_worst.png")
    print(np.mean(a1), np.mean(a2))
    t_stat, p_value = stats.wilcoxon(vss, vs_n,
                                     alternative='greater')
    print(p_value, p_value < alpha, np.mean(np.subtract(vss, vs_n)))
    plt.figure()
    plt.plot(np.subtract(vss, vs_n))
    plt.savefig("test_accuracies2.png")


def evaluateE1_E2(fol_result):
    # 1. Load the data
    # Assuming the files contain lists of labels (e.g., [0, 1, 1, 0...])
    if False:
        with open(os.path.join(fol_result, 'oecd.fewshot_selected.confusion.json'), 'r') as f:
            cm_s = np.array(json.load(f))

        with open(os.path.join(fol_result, 'oecd.fewshot_not_selected.confusion.json'), 'r') as f:
            cm_ns = np.array(json.load(f))
    else:
        with open(os.path.join(fol_result, 'selected.fewshot_selected.confusion.json'), 'r') as f:
            cm_s = np.array(json.load(f))

        with open(os.path.join(fol_result, 'train.not_selected.confusion.json'), 'r') as f:
            cm_ns = np.array(json.load(f))
        with open('third_experiment.json', 'r') as f:
            cm_s, cm_ns = [np.array(x) for x in json.load(f)]
    print(cm_s.shape, cm_ns.shape)
    results_all = []
    names = []

    def print_stats(y_true_o, y_pred_o, name):
        inds = np.arange(len(y_true_o))
        np.random.seed(42)
        np.random.shuffle(inds)
        inds02 = [[i for i in inds if y_true_o[i] == 0], [i for i in inds if y_true_o[i] == 2]]
        minl = min([len(x) for x in [inds02[0], inds02[1]]])
        inds02 = sorted(chain.from_iterable([x[:minl] for x in inds02]))
        y_true02 = np.array(y_true_o)[inds02]
        y_pred02 = np.array(y_pred_o)[inds02]
        y_pred02 = np.where(y_pred02 == 1, 0, y_pred02)
        y_true02 = y_true02 // 2
        y_pred02 = y_pred02 // 2
        inds_stratified = [[i for i in inds if y_true_o[i] == 0], [i for i in inds if y_true_o[i] == 1],
                           [i for i in inds if y_true_o[i] == 2]]
        minl = min([len(x) for x in [inds_stratified[0], inds_stratified[1], inds_stratified[2]]])
        inds_selected = sorted(chain.from_iterable([x[:minl] for x in inds_stratified]))
        y_true = np.array(y_true_o)[inds_selected]
        y_pred = np.array(y_pred_o)[inds_selected]

        accuracy = accuracy_score(y_true, y_pred)

        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')

        names.append(name)
        results_all.append({"Accuracy": accuracy, "Precision0v2": precision_score(y_true02, y_pred02, average='binary'),
                            "Recall0v2": recall_score(y_true02, y_pred02, average='binary'),
                            "Quadratic Weighted Kappa": qwk})

    def tol(cm):
        ds1 = []
        ds2 = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ds1.extend([i] * cm[i, j])
                ds2.extend([j] * cm[i, j])
        return ds1, ds2

    print(np.sum(np.diagonal(cm_s)), np.sum(cm_s))
    print(np.sum(np.diagonal(cm_ns)), np.sum(cm_ns))
    print("----")
    print_stats(*tol(cm_s), name="Selected on Oecd")
    print("\n----")
    print_stats(*tol(cm_ns), name="All on Oecd")

    st = pd.DataFrame().from_records(results_all, index=names).to_latex(
        float_format="%.03f")
    with open("table.txt", "w") as f:
        f.write(st)

    with open('rtfewshot_dataset_not_selected.json', 'r') as f:
        preds_ns = np.array(json.load(f))

    with open("subsample.json", 'r') as f:
        selected_ids = set([x["id"] for x in np.array(json.load(f))])

    print_stats([x["correct_label"] for x in preds_ns if x["id"] in selected_ids],
                [x["label"] for x in preds_ns if x["id"] in selected_ids], "All on Selected")
    print_stats([x["correct_label"] for x in preds_ns], [x["label"] for x in preds_ns], "All on All")

    st = pd.DataFrame().from_records(results_all, index=names).to_latex(
        float_format="%.03f")
    with open("table.txt", "w") as f:
        f.write(st)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--experiment", type=str, default="E1")
    argparser.add_argument("--fol_subsample", type=str, default="subsample_test_accuracies")
    argparser.add_argument("--fol_non_subsample", type=str, default="non_subsample_test_accuracies")
    argparser.add_argument("--fol_result", type=str, default="./")
    args = argparser.parse_args()
    if args.experiment == "E3":
        evaluateE3(args.fol_subsample, args.fol_non_subsample)
    else:
        evaluateE1_E2(args.fol_result)
