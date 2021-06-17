# sts_eval: Easy Evaluation of Semantic Textual Similarity for Neural Language Models

This is a small framework aimed to make easy the evaluation of Language Models with the STS Benchmark as well as other task specific datasets. Our goal is to make easy to compare different versions of LMs as you work to improve them for a specific STS task.

The framework wraps models from different sources and runs the selected evaluation with them, producing a standarized JSON output.

Models can be sourced from:

- [Tensorflow Hub, Embedding models](https://tfhub.dev/s?module-type=text-embedding)
- [SBERT Sentence Transformers](https://www.sbert.net/docs/pretrained_models.html)
- [HuggingFace models](https://huggingface.co/models) - IN DEVELOPMENT!!!

Output JSON example:

```
{
    "class": "SentenceTransformerSTSEvaluator",
    "model_url": "stsb-roberta-base-v2",
    "data_filename": "data/stsbenchmark/sts-dev.csv",
    "used_minimal_normalization": null,
    "metric": "cosine",
    "scaled_scores": null,
    "evaluation": {
        "pearson": {
            "r": 0.8872942593119845,
            "p-value": 0.0
        },
        "spearman": {
            "rho": 0.8861646506975909,
            "p-value": 0.0
        }
    },
    "timestamp": "2021-06-14 23:13:05",
    "metadata": [],
    "tag": "stsb--roberta_base_v2-cosine",
    "benchmark": "stsbenchmark"
}
```

### Main Goal: Extension to other evaluation datasets

The main goal of this framework is to help in the evaluation of Language Models for other context-specific tasks. 

**TODO** Example with STS for product names or ad titles

### Evaluation Results

Check [this](./LanguageModelsSTSEvaluation.ipynb) for the current results of evaluating several LMs in the standard datasets and in the context-specific example. This results closely resembles the ones published in [PapersWithCode](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark) and [SBERT Pretrained Models](https://www.sbert.net/docs/pretrained_models.html)

![STSBenchmark results](img/stsb-spearman.png)

![STS-2017-es-es results](img/stses-spearman.png)


## Usage

Clone this repo, then you can build a docker image with all dependecies already integrated from [this repository](https://github.com/eduardofv/ai-denv) and run it inside here. Use the main script [sts_evaluation.py](sts_evaluation.py) with the following parameters:

- Evaluator type: 
	- `tfhub` for Tensorflow Hub models that can embed strings directly
	- `sent` for SentenceTransormers models
	- `hf` for HuggingFace models that can embed strings as AutoML models
 
- Model: use the URL, identifier or directory as required by the model.
- Benchmark: See below for available datasets
	- `stsbenchmark`
	- `sts-es` 2017 Spanish to Spanish 
- *Optional*, similarity metric: `cosine` (default) or `euclidean`. Euclidean similarity defined as `1 / (1 + euclidean_distance)`
- *Optional*, tag: any tag that may help you to identify this particular run.

```
tf-docker /root > python sts_evaluation.py sent stsb-roberta-base-v2 stsbenchmark cosine 2> /dev/null 1> results/stsb--stsb-roberta-base-v2.json cosine 
```

## Datasets

### STS Benchmark

http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark

https://paperswithcode.com/sota/semantic-textual-similarity-on-sts-benchmark

### SemEval 2017

For Spanish monolingual texts:

https://alt.qcri.org/semeval2017/task1/

### SentEval (Facebook)

https://github.com/facebookresearch/SentEval

