from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import utils
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
logger = logging.getLogger(__name__)


class EmbeddingMinMaxFeaturizer(Featurizer):
    """Sentence Embedding Featurizer

    Sentence embedding is calculated as weighted average of word embedding by TF-IDF

    When there is no TF-IDF available, just compute sentence embedding as simple average of word embeddings 

    Word2Vec model shall be trained on lemmas

    TF using frequency instead of real counts.
    """

    name = "intent_featurizer_min_max_embedding"

    provides = ["text_features"]

    requires = ["word_vectors"]

    defaults = {}

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy"]

    def __init__(self, component_config=None):
        """Construct a new sentence embedding featurizer:
            - load a trained word2vec model"""
        super(EmbeddingMinMaxFeaturizer, self).__init__(component_config)

    def get_embedding(self, example):
        '''
        Sentence embedding as tf-idf weighted average of word embedding. 
        OOV words regard as 0 vectors.
        '''
        wv = example.get("word_vectors")
        min_wv = np.min(wv, axis=1)
        max_wv = np.max(wv, axis=1)
        eb = np.hstack((min_wv, max_wv))
        return eb

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """First, copmute the IDF of given corpus (training data), 
        and the IDF shall be persisted later.
        Then, for each training piece:
         - Retrieve "word_vectors";
         - Compute TF-IDF;
         - Compute sentence embedding and set as text feature"""

        for example in training_data.intent_examples:
            # create bag for each example
            sentence_embedding = self.get_embedding(example)
            example.set("text_features",
                        self._combine_with_existing_text_features(example, sentence_embedding))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        sentence_embedding = self.get_embedding(message)
        message.set("text_features",
                    self._combine_with_existing_text_features(message, sentence_embedding))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""
        ## TODO: how to persist? read about pickle.
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"minmax_sentence_embedding_file": self.name + ".pkl"}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> CountVectorsFeaturizer
        ## TODO:
        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("minmax_sentence_embedding_file"):
            file_name = meta.get("minmax_sentence_embedding_file")
            sentence_embedding_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(sentence_embedding_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return EmbeddingMinMaxFeaturizer(meta)
