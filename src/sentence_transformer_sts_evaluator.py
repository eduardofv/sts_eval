from sentence_transformers import SentenceTransformer
from src.language_model_sts_evaluator import BaseSTSEvaluator

class SentenceTransformerSTSEvaluator(BaseSTSEvaluator):
    """
    Wraps a SentenceTransformer LM for STS Evaluation
    https://www.sbert.net/index.html
    """
    def __build_model__(self):
        self.model = SentenceTransformer(self.model_url)

    def embed(self, texts):
        return self.model.encode(texts)
