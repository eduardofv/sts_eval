"""
Perform a STS evaluation on our predefined datasets

This was adapted from analysis/STS%20Evaluation%20for%20Language%20Models.ipynb
"""

import json
import logging
import os
import sys

logging.basicConfig(level=logging.CRITICAL)

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    logging.warning("***\nWARNING: set_memory_growth to true. Experimental\n***")
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

#Evaluators for different Language Model Frameworks
import src.language_model_sts_evaluator as stsev 
from src.basic_tfhub_sts_evaluator import BasicTFHubSTSEvaluator
from src.sentence_transformer_sts_evaluator import SentenceTransformerSTSEvaluator
from src.hf_sts_evaluator import HuggingfaceAutoMLSTSEvaluator


benchmarks = {
    'stsbenchmark': {
        'filename': "data/stsbenchmark/sts-dev.csv",
        'loader': stsev.load_stsbenchmark
    },
    'sts-es': {
        'filename': "data/stsbenchmark/STS2017.track3.es-es.tsv",
        'loader': stsev.load_sts2017es
    }    
}

metrics = ["cosine", "euclidean"]

evaluators = {
    'tfhub': BasicTFHubSTSEvaluator,
    'sent': SentenceTransformerSTSEvaluator,
    'hf': HuggingfaceAutoMLSTSEvaluator
}

def add_benchmark(name, config):
    benchmarks[name] = config 

def perform_evaluation(evaluator_type, url, benchmark, metric="cosine", tag=""):
    """Executes the evaluation process for the corresponding evaluator"""
    assert benchmark in benchmarks.keys()
    assert evaluator_type in evaluators.keys()
    assert metric in metrics

    curr_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(curr_dir, benchmarks[benchmark]['filename'])
    #url is a list if both URLs to model and to tokenizer are needed
    evaluator = evaluators[evaluator_type]
    if isinstance(url, list):
        ev = evaluator(*url) 
    else:
        ev = evaluator(url)
    ev.perform_sts_evaluation(
        filename,
        benchmarks[benchmark]['loader'],
        metric=metric
    )
    ret = ev.data()
    ret['tag'] = tag
    ret['benchmark'] = benchmark
    return ret


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"""
Perform a STS Evaluation for a Language model
Usage: sts_evaluation.py evaluator model benchmark [metric] [tag]

\tevaluator:    {list(evaluators.keys())}
\tmodel:        model name, directory or URL
\tbenchmark:    {list(benchmarks.keys())}
\tmetric:       {metrics}
\ttag:          optional identifying tag
""")
    else:
        print(json.dumps(perform_evaluation(*sys.argv[1:]), indent=4))
