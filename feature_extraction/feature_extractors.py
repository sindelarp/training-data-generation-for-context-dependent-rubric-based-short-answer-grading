import re
import string
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Optional

import nltk
import torch
import torch.nn.functional
from nltk import ngrams, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, FeatureExtractionPipeline

nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download("averaged_perceptron_tagger_eng")

STOPWORDS = set(stopwords.words("english"))
PUNCT = set(string.punctuation)


def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return [t for t in word_tokenize(text) if t not in PUNCT and t not in STOPWORDS]


def ngram_set(ngram_set: List[str], n: int) -> set:
    return set(ngrams(ngram_set, n))


def resolve_path(data, path, delimiter="/"):
    keys = path.strip(delimiter).split(delimiter)
    return reduce(lambda d, key: d[key], keys, data)


class FeatureExtractor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def extract_features(self, data: list[dict]) -> list[float]:
        pass


class EmbeddingCosineSimilarity(FeatureExtractor):
    def __init__(
            self, model_name: str, field1: str, field2: str, device: int = 0
    ):
        self._model_name = model_name
        self._field1 = field1
        self._field2 = field2
        self._device = device

    @property
    def name(self) -> str:
        return f"{self._model_name}_{self._field1}_{self._field2}_cosine_similarity"

    def _get_embedding(
            self, text: str, extractor: FeatureExtractionPipeline
    ) -> torch.Tensor:
        raw_output = extractor(text, return_tensors=True)
        # pyrefly: ignore[missing-attribute]
        sentence_embedding = raw_output.mean(
            dim=1
        ).squeeze()
        return sentence_embedding

    def extract_features(self, data: list[dict]) -> list[float]:
        extractor = pipeline(
            task="feature-extraction", model=self._model_name, device=self._device
        )
        features = []
        for item in data:
            embedding1 = self._get_embedding(resolve_path(item, self._field1), extractor)
            embedding2 = self._get_embedding(resolve_path(item, self._field2), extractor)
            cosine_sim = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0),
            ).item()
            features.append(cosine_sim)
        return features


class TokenLengthExtractor(FeatureExtractor):
    def __init__(self, field: str):
        self._field = field

    @property
    def name(self) -> str:
        return f"{self._field}_length"

    def extract_features(self, data: List[dict]) -> List[float]:
        features = []
        for item in data:
            text = preprocess(resolve_path(item, self._field))
            tokens = tokenize(text)
            features.append(float(len(tokens)))
        return features


class LexicalDensityExtractor(FeatureExtractor):
    def __init__(self, field: str):
        self._field = field

    @property
    def name(self) -> str:
        return f"{self._field}_lexical_density"

    def extract_features(self, data: List[dict]) -> List[float]:
        features = []
        for item in data:
            text = preprocess(resolve_path(item, self._field))
            tokens = tokenize(text)
            if not tokens:
                features.append(0.0)
                continue

            pos = pos_tag(tokens)
            # Count Nouns, Verbs, Adjectives, Adverbs
            content_count = sum(
                1 for _, tag in pos if tag.startswith(("N", "V", "J", "R"))
            )
            features.append(content_count / len(tokens))
        return features


class QuestionTypeExtractor(FeatureExtractor):
    QUESTION_TYPE_MAP = {
        "who": 0,
        "what": 1,
        "when": 2,
        "where": 3,
        "why": 4,
        "how": 5,
        "which": 6,
        "other": 7,
    }

    def __init__(self, field: str = "question"):
        self._field = field

    @property
    def name(self) -> str:
        return f"{self._field}_type_label"

    def extract_features(self, data: List[dict]) -> List[float]:
        features = []
        for item in data:
            q_text = preprocess(resolve_path(item, self._field))
            label = self.QUESTION_TYPE_MAP["other"]
            for start_word, code in self.QUESTION_TYPE_MAP.items():
                if q_text.startswith(start_word):
                    label = code
                    break
            features.append(float(label))
        return features


class NgramOverlapExtractor(FeatureExtractor):

    def __init__(self, field1: str, field2: str, n: int = 1, metric: str = "jaccard"):
        self._field1 = field1
        self._field2 = field2
        self._n = n
        self._metric = metric  # 'jaccard', 'precision', 'recall'

    @property
    def name(self) -> str:
        return f"{self._metric}_{self._n}gram_{self._field1}_{self._field2}"

    def extract_features(self, data: List[dict]) -> List[float]:
        features = []
        for item in data:
            tokens1 = tokenize(preprocess(resolve_path(item, self._field1)))
            tokens2 = tokenize(preprocess(resolve_path(item, self._field2)))

            if self._n == 1:
                set1, set2 = set(tokens1), set(tokens2)
            else:
                set1 = ngram_set(tokens1, self._n)
                set2 = ngram_set(tokens2, self._n)

            if len(set1) == 0 and len(set2) == 0:
                features.append(0.0)
                continue

            intersection = len(set1 & set2)

            if self._metric == "jaccard":
                union = len(set1 | set2)
                features.append(intersection / union if union > 0 else 0.0)
            elif self._metric == "recall":
                features.append(intersection / len(set2) if len(set2) > 0 else 1.0)
            elif self._metric == "precision":
                features.append(intersection / len(set1) if len(set1) > 0 else 0.0)
            else:
                raise ValueError(f"Unsupported metric: {self._metric}")
        return features


class ConditionalOverlapExtractor(FeatureExtractor):

    def __init__(
            self, field1: str, field2: str, field3: str, n: int = 2
    ):
        self._field1 = field1
        self._field2 = field2
        self._field3 = field3
        self._n = n

    @property
    def name(self) -> str:
        return f"recall_{self._n}gram_{self._field1}_overlap_with_{self._field2}_minus_{self._field3}"

    def extract_features(self, data: List[dict]) -> List[float]:
        features = []
        for item in data:
            tok_1 = tokenize(preprocess(resolve_path(item, self._field1)))
            tok_2 = tokenize(preprocess(resolve_path(item, self._field2)))
            tok_3 = tokenize(preprocess(resolve_path(item, self._field3)))

            if self._n == 1:
                set_1, set_2, set_3 = set(tok_1), set(tok_2), set(tok_3)
            else:
                set_1 = ngram_set(tok_1, self._n)
                set_2 = ngram_set(tok_2, self._n)
                set_3 = ngram_set(tok_3, self._n)

            diff_set = set_2 - set_3

            if len(diff_set) == 0:
                features.append(1.0)
                continue

            overlap = len(set_1 & diff_set)
            features.append(overlap / len(diff_set))

        return features


class TfidfCosineSimilarityExtractor(FeatureExtractor):
    def __init__(
            self, field1: str, field2: str, training_corpus: Optional[List[str]] = None
    ):
        self._field1 = field1
        self._field2 = field2
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fitted = False

        if training_corpus:
            self.fit(training_corpus)

    @property
    def name(self) -> str:
        return f"tfidf_cosine_{self._field1}_{self._field2}"

    def fit(self, corpus: List[str]):
        try:
            self._vectorizer.fit([preprocess(t) for t in corpus])
        except ValueError as e:
            print([(t, preprocess(t)) for t in corpus][:10])
            raise e
        self._is_fitted = True

    def extract_features(self, data: List[dict]) -> List[float]:
        if not self._is_fitted:
            all_text = [preprocess(resolve_path(d, self._field1)) for d in data] + [
                preprocess(resolve_path(d, self._field2)) for d in data
            ]
            self.fit(all_text)

        features = []
        for item in data:
            v1 = self._vectorizer.transform([preprocess(resolve_path(item, self._field1))])
            v2 = self._vectorizer.transform([preprocess(resolve_path(item, self._field2))])

            sim = cosine_similarity(v1, v2)[0][0]
            features.append(float(sim))
        return features


class KneserNeyPerplexityExtractor(FeatureExtractor):
    def __init__(
            self,
            field: str,
            training_corpus: Optional[List[str]] = None,
            order: int = 3,
    ):
        self._field = field
        self._order = order
        self._model = KneserNeyInterpolated(order)
        self._is_fitted = False

        if training_corpus:
            self.fit(training_corpus)

    @property
    def name(self) -> str:
        return f"kn_perplexity_{self._field}"

    def fit(self, corpus: List[str]):
        tokenized_corpus = [tokenize(preprocess(text)) for text in corpus]
        train_data, vocab = padded_everygram_pipeline(self._order, tokenized_corpus)
        self._model.fit(train_data, vocab)
        self._is_fitted = True

    def extract_features(self, data: List[dict]) -> List[float]:
        if not self._is_fitted:
            raise ValueError(
                "KneserNeyPerplexityExtractor must be fitted with a training corpus before extraction."
            )

        features = []
        for item in data:
            tokens = tokenize(preprocess(resolve_path(item, self._field)))
            if len(tokens) < self._order:
                features.append(0.0)
            else:
                try:
                    test_ngrams = list(ngrams(tokens, self._order))
                    perp = self._model.perplexity(test_ngrams)
                    features.append(perp)
                except:
                    features.append(0.0)
        return features
