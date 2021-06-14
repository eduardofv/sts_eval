import tensorflow_hub as hub
from src.language_model_sts_evaluator import BaseSTSEvaluator

class BasicTFHubSTSEvaluator(BaseSTSEvaluator):
    """Wraps a TFHub LM with no preprocessing necessary for STS Evaluation"""

    def __build_model__(self):
        self.model = hub.load(self.model_url)

    def embed(self, texts):
        return self.model(texts).numpy()

