"""
Huggingface-models-based STS Evaluator
Based on
analysis/crear_emb_from_ont_names-v5.ipynb
"""

import numpy as np
import tensorflow as tf

from transformers import TFBertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

import src.language_model_sts_evaluator as stsev


class HuggingfaceBertSTSEvaluator(stsev.BaseSTSEvaluator):
    
    def __build_model__(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.model_url)
        self.model = TFBertModel.from_pretrained(self.model_url)


    def __bert__(self, tokens, layers_to_average=2):
        outputs = self.model(tokens, output_hidden_states=True)
        hidden_states = np.array(outputs.hidden_states[-layers_to_average:])
        hidden_states_ft = np.transpose(hidden_states, (1, 2, 0, 3))
        emb = hidden_states_ft.mean(axis=2).mean(axis=1)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms


    def embed(self, texts, batches=10):
        emb = None
        for batch in np.array_split(texts, batches):
            tokens = self.tokenizer(list(batch), 
                                    padding=True, return_tensors="tf")
            if emb is None:
                emb = self.__bert__(tokens)
            else:
                emb = tf.concat([emb, self.__bert__(tokens)], axis=0)
        return emb


class HuggingfaceAutoMLSTSEvaluator(stsev.BaseSTSEvaluator):

    def __build_model__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_url)
        self.model = AutoModel.from_pretrained(self.model_url)


    def embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, 
                truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            emb = model_output[0][:,0] 
        return emb
