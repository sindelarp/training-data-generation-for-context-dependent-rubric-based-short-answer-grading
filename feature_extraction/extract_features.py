import argparse
import json

import numpy as np
from dotenv import load_dotenv

from feature_extractors import *


def main(inp):
    torch.backends.cuda.matmul.allow_tf32 = True
    with open(inp, "r", encoding="utf-8") as f:
        data = json.load(f)
    data_o = [x for x in data if x["lang"] == "en"]
    np.random.seed(100)
    np.random.shuffle(data)
    data = []
    for label in set([x["label"] for x in data_o]):
        data += [x for x in data_o if x["label"] == label][:100]
    feature_extractors = [
        EmbeddingCosineSimilarity(
            model_name="BAAI/bge-m3",
            field1="input/context",
            field2="input/question",
            device=0,
        ),
        EmbeddingCosineSimilarity(
            model_name="BAAI/bge-m3",
            field1="input/context",
            field2="input/answer",
            device=0,
        ),
        EmbeddingCosineSimilarity(
            model_name="BAAI/bge-m3",
            field1="input/rubrics/FC",
            field2="input/answer",
            device=0,
        ),
        NgramOverlapExtractor("input/rubrics/FC", "input/answer", n=2, metric="recall"),
        NgramOverlapExtractor("input/rubrics/NC", "input/answer", n=2, metric="recall"),
        TokenLengthExtractor(field="input/answer"),
        LexicalDensityExtractor(field="input/answer"),
        QuestionTypeExtractor(field="input/question"),
        NgramOverlapExtractor("input/question", "input/answer", n=1, metric="jaccard"),
        NgramOverlapExtractor("input/context", "input/answer", n=1, metric="jaccard"),
        NgramOverlapExtractor("input/question", "input/answer", n=2, metric="recall"),
        NgramOverlapExtractor("input/context", "input/answer", n=2, metric="recall"),
        ConditionalOverlapExtractor(
            "input/context", "input/answer", "input/question", n=2
        ),
        TfidfCosineSimilarityExtractor("input/question", "input/answer"),
        TfidfCosineSimilarityExtractor("input/context", "input/answer"),
        NgramOverlapExtractor(
            "input/question", "input/answer", n=1, metric="precision"
        ),
        NgramOverlapExtractor("input/question", "input/answer", n=1, metric="recall"),
        NgramOverlapExtractor("input/context", "input/answer", n=1, metric="recall"),
        ConditionalOverlapExtractor(
            "input/context", "input/answer", "input/question", n=1
        ),
    ]

    features = {
        extractor.name: extractor.extract_features(data)
        for extractor in feature_extractors
    }
    feature_rows = [dict(zip(features, t)) for t in zip(*features.values())]
    for x, y in zip(data, feature_rows):
        y["label"] = x["label"]
        y["lang"] = x["lang"]
    with open(inp + ".features.json", "w", encoding="utf-8") as f:
        json.dump(feature_rows, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    load_dotenv()
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--inp", type=str)
    args = argparser.parse_args()
    main(args.inp)
