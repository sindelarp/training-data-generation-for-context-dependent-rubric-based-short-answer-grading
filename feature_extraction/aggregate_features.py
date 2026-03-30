import json

import sklearn
from dotenv import load_dotenv

from feature_aggregators import *


def main(args):
    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)

    aggregate_features_by_class_and_cluster = {}
    filters = {"all": lambda x: True, "0": lambda x: x["label"] == 0, "1": lambda x: x["label"] == 1,
               "2": lambda x: x["label"] == 2}
    for name, filterf in filters.items():
        data1 = [x for x in data if filterf(x)]
        ks = [x for x in data1[0].keys() if x != "label"]

        np.random.seed(42)
        np.random.shuffle(data1)
        data1 = data1[:5000]
        features = np.array([[item[k] for k in ks] for item in data1])
        scaler = sklearn.preprocessing.MinMaxScaler()
        features = scaler.fit_transform(features)
        num_clusters = 10
        if num_clusters > 1:
            clusters = sklearn.cluster.KMeans(n_clusters=num_clusters, n_init=50).fit_predict(features)
            print(sklearn.metrics.silhouette_score(features, clusters),
                  sklearn.metrics.silhouette_score(features, np.random.randint(0, 8, clusters.shape)))
        else:
            clusters = [0] * len(data1)
        aggregate_features_by_class_and_cluster[name] = [None] * num_clusters
        for i in np.unique(clusters):
            i = int(i)
            data2 = [x for c, x in zip(clusters, data1) if c == i]
            feature_aggregators: list[FeatureAggregator] = []

            for feature in data2[0].keys():
                feature_aggregators.append(MeanAggregator(feature))
                feature_aggregators.append(PercentileAggregator(feature, [25, 50, 75]))
            for item in data2:
                for aggregator in feature_aggregators:
                    aggregator.add(item)

            aggregate_features = {
                aggregator.name: aggregator.aggregate()
                for aggregator in feature_aggregators
            }
            aggregate_features_by_class_and_cluster[name][i] = aggregate_features

    with open(args.inp + ".aggregate.json", "w", encoding="utf-8") as f:
        json.dump(aggregate_features_by_class_and_cluster, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    load_dotenv()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=str, default="complete_generated_data.features.json")
    args = parser.parse_args()
    main(args)
