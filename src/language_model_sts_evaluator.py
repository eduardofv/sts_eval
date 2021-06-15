"""
Language Model wrappers for STS Benchmark-like evaluation
"""

import re
import datetime as dt
import numpy as np
import pandas as pd
import sklearn
import scipy

import tensorflow as tf 
import tensorflow_hub as hub
import tensorflow_text

from sentence_transformers import SentenceTransformer


def load_stsbenchmark(dataset_filename):
    """Loads the STSBenchmark dataset"""
    lines = open(dataset_filename).readlines()
    sts = [l.strip().split("\t")[:7] for l in lines]
    sentence_1 = [e[5] for e in sts]
    sentence_2 = [e[6] for e in sts]
    dev_scores = [float(e[4]) for e in sts]    
    return (sentence_1, sentence_2, dev_scores)


def load_sts2017es(dataset_filename):
    """Loads the prebuilt STS2017 es-es dataset"""
    lines = open(dataset_filename).readlines()
    sts = [l.strip().split("\t") for l in lines]
    sentence_1 = [e[0] for e in sts]
    sentence_2 = [e[1] for e in sts]
    dev_scores = [float(e[2]) for e in sts]    
    return (sentence_1, sentence_2, dev_scores)


class BaseSTSEvaluator():
    """Base class to derive models to be evaluated using STS like benchmarks"""

    def __init__(self, model_url):
        self.model_url = model_url
        self.filename = None
        self.metric = None
        self.used_minimal_normalization = None
        self.scaled_scores = None
        self.evaluation = None
        self.timestamp = None
        self.metadata = []
        self.__build_model__()


    def __build_model__(self):
        """Setup the Language Model"""
        raise NotImplementedError


    def data(self):
        """
        Returns a dict representing the object
        On subclasses, override and call this to add new elements
        """
        data = {
            'class': str(type(self).__name__),
            'model_url': self.model_url,
            'data_filename': self.filename,
            'used_minimal_normalization': self.used_minimal_normalization,
            'metric': self.metric,
            'scaled_scores': self.scaled_scores,
            'evaluation': self.evaluation,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            #'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
        return data


    def __str__(self):
        return str(self.data())


    def embed(self, texts):
        """Get embeddings from a group of texts"""
        raise NotImplementedError


    def calculate_similarities(self, sentences_a, sentences_b, 
            metric="euclidean", minimal_normalization=True):
        """Return similarities between two lists of sentences"""
        self.metric = metric
        self.minimal_normalization = minimal_normalization

        if minimal_normalization:
            sentences_a = [re.sub(r"[\t\n,]", " ", e.lower())
                for e in sentences_a]
            sentences_b = [re.sub(r"[\t\n,]", " ", e.lower())
                for e in sentences_b]
        sentences_a_emb = self.embed(sentences_a)
        sentences_b_emb = self.embed(sentences_b)

        if metric == 'euclidean':
            distances = [np.linalg.norm(v[1]-v[0])
                for v in zip(sentences_a_emb, sentences_b_emb)]
            #similarities = max(distances) - distances
            similarities = 1 / (1 + np.array(distances))
        elif metric == "cosine":
            similarities = [np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(a)) 
                            for a,b in zip(sentences_a_emb, sentences_b_emb)]
        else:
            raise ValueError(f"Incorrect metric {metric}")

        return similarities


    def scale_scores(self, real_scores, similarities):
        """
        Linearly scale scores (or similarities) to be within the range
        of real_scores. For STS it's [0;5]
        """
        self.scaled_scores = True
        return sklearn.preprocessing.minmax_scale(similarities, 
            feature_range=(min(real_scores), max(real_scores)))


    def evaluate_correlation(self, real_scores, scores):
        """Returns a dict with both Pearson and Spearman evaluations"""
        pearson = scipy.stats.pearsonr(scores, real_scores)
        pearson = {'r': pearson[0], 'p-value': pearson[1]}
        spearman = scipy.stats.spearmanr(scores, real_scores)
        spearman = {'rho': spearman[0], 'p-value': spearman[1]}
        return {'pearson':pearson, 'spearman':spearman}


    def perform_sts_evaluation(self, filename, 
            loader, metric="cosine", scale_scores=False,
            minimal_normalization=True):
        """Complete STS Jobtitile evaluation"""

        self.filename = filename
        self.timestamp = dt.datetime.now()
        sentences_1, sentences_2, dev_scores = loader(filename)
        similarities = self.calculate_similarities(sentences_1, sentences_2,
            metric=metric, minimal_normalization=minimal_normalization)
        if scale_scores:
            similarities = self.scale_scores(dev_scores, similarities)
        self.scores = similarities
        self.evaluation = self.evaluate_correlation(dev_scores, similarities)
        return self.evaluation
