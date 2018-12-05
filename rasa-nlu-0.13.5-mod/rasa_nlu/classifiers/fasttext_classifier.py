from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
from tqdm import tqdm

import typing
from typing import List, Text, Any, Optional, Dict

from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.model import Metadata
    from rasa_nlu.training_data import Message


try:
    import fastText as ft
except ImportError:
    ft = None


try:
    import tensorflow as tf
except ImportError:
    tf = None


class FasttextIntentClassifier(Component):
    """Intent classifier using Python implementation of Facebook fasttext algorithm.
    https://github.com/facebookresearch/fastText/tree/master/python
    """
    name = "intent_classifier_fasttext"

    provides = ["intent", "intent_ranking"]

    requires = []

    defaults = {
       "subwords": "model_sub_wiki.vec"
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["fastText"]

    def _load_nn_architecture_params(self):
        self.num_hidden_layers_a = self.component_config['num_hidden_layers_a']


    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 model=None,
                 wv=''):
        self.model = model
        self.wv = wv
        # type: (...) -> None
        """Declare instant variables with default values"""
        super(FasttextIntentClassifier, self).__init__(component_config)

        # nn architecture parameters
 
    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], **Any) -> None
        """Train the fasttext intent classifier on a data set."""

        def write_to_file(filename, text):
            # type: (Text, Text) -> None
            """Write a text to a file."""
            with io.open(filename, 'w', encoding="utf-8") as f:
                f.write(str(text))
        # prepare .txt for fasttext
        data_file = "training_data.txt"
        data = ''
        for msg in training_data.intent_examples:
            data += "__label__" + msg.get("intent") + ' '
            data += msg.text + "\n"
        write_to_file("fasttext_training.txt", data)

        # train the classifier
        self.model = ft.train_supervised('fasttext_training.txt', dim=300, pretrainedVectors=self.wv)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if not self.model:
            logger.error("There is no trained classifier")

        else:
            # get features (bag of words) for a message
            preds, probs = self.model.predict([message.text], 
                                              k=len(self.model.get_labels()))
            preds = [label[9:] for label in preds[0]]
            probs = probs[0]
            
            for i, (prob, pred) in enumerate(sorted(zip(probs, preds), reverse=True)):
                if i >= INTENT_RANKING_LENGTH:
                    break
                intent['name'] = pred
                intent['confidence'] = prob
                intent_ranking.append(intent.copy())
                
            idx = np.argmax(probs)
            intent['name'] = preds[idx]
            intent['confidence'] = probs[idx]  
        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)
    
    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> GoogleNewsWV
        component_conf = cfg.for_component(cls.name, cls.defaults)
        subwords_model_name = component_conf.get("subwords")

        return FasttextIntentClassifier(component_conf, wv=subwords_model_name)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        ft_model_path = os.path.join(model_dir, self.name)
        self.model.save_model(ft_model_path)

        return {"fasttext_classifier_model": self.name}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingIntentClassifier

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("fasttext_classifier_model"):
            file_name = meta.get("fasttext_classifier_model")
            fasttext_classifier_model = os.path.join(model_dir, file_name)
            return cls(component_config=meta, 
                       model=ft.load_model(fasttext_classifier_model))
        else:
            logger.warning("Failed to load fasttext model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
