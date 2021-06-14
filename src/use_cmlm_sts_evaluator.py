import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

from src.language_model_sts_evaluator import *

class USECMLMSTSEvaluator(BaseSTSEvaluator):
    """
    Universal Sentence Encoder Conditional Masked Language Model
    STS Evaluator
    https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1
    """

    def __init__(self, model_url, preprocessor_url):
        self.preprocessor_url = preprocessor_url
        BaseSTSEvaluator.__init__(self, model_url)
    
    def __build_model__(self):
        self.preprocessor = hub.KerasLayer(self.preprocessor_url)
        self.model = hub.KerasLayer(self.model_url)

    def embed(self, texts):
        texts = tf.constant(texts)
        embeddings = self.model(self.preprocessor(texts))['default']
        return embeddings
