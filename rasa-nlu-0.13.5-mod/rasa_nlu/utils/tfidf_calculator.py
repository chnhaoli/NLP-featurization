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
import numpy as np
logger = logging.getLogger(__name__)


class TfidfCalculator(Component):
    """Sentence Embedding Featurizer

    Sentence embedding is calculated as weighted average of word embedding by TF-IDF

    When there is no TF-IDF available, just compute sentence embedding as simple average of word embeddings 

    Word2Vec model shall be trained on lemmas

    TF using frequency instead of real counts.
    """

    name = "tfidf_calculator"

    provides = ["tfidf"]

    requires = ["token_text"]

    defaults = {}

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy"]

    def __init__(self, component_config=None, idf=None, zero_idf=None):
        """Construct a new sentence embedding featurizer:
            - load a trained word2vec model"""
        self.idf = idf
        self.zero_idf = zero_idf
        super(TfidfCalculator, self).__init__(component_config)

    @staticmethod
    def idf_lookup_table(examples):
        '''
        Parameter:
        examples: training_data.intent_examples
        Return:
        dictionary: {word : idf}
        '''
        vocab = set([word for example in examples for word in example.get("token_text")])
        df_table = {word : 0 for word in vocab}
        for example in examples:
            for term in set(example.get("token_text")):
                df_table[term] += 1
        return {word : np.log(len(examples)/(df_table[word]+1)) for word in vocab}, np.log(len(examples))

    @staticmethod
    def tf_lookup_table(sentence):
        '''
        Parameter:
        sentence: token_text
        Return:
        dictionary: {word : tf}
        '''
        vocab = set(sentence)
        tf_table = {word : 0 for word in vocab}
        for word in sentence:
            tf_table[word] += 1/len(sentence)
        return tf_table

    def get_tfidf(self, sentence):
        '''
        Sentence embedding as tf-idf weighted average of word embedding. 
        OOV words regard as 0 vectors.
        '''
        tfidf = []
        tf = self.tf_lookup_table(sentence)
        for word in sentence:
            try:
                # word in sentence can be found in w2v model, i.e., not OOV words
                tfidf.append(tf[word] * self.idf.get(word, self.zero_idf))
                    #print('[1] word with TFIDF')
            except:
                raise LookupError("IDF lookup table not available")
        return np.array(tfidf).reshape(1,len(sentence))

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """First, copmute the IDF of given corpus (training data), 
        and the IDF shall be persisted later.
        Then, for each training piece:
         - Compute TF-IDF;
         - Compute sentence embedding and set as text feature"""

        # compute IDF
        self.idf, self.zero_idf = self.idf_lookup_table(training_data.intent_examples)

        for example in training_data.intent_examples:
            # create bag for each example
            example.set("tfidf", self.get_tfidf(example.get("token_text")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tfidf", self.get_tfidf(message.get("token_text")))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""
        ## TODO: how to persist? read about pickle.
        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"tfidf_file": self.name + ".pkl"}

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

        if model_dir and meta.get("tfidf_file"):
            file_name = meta.get("tfidf_file")
            sentence_embedding_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(sentence_embedding_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return TfidfCalculator(meta)
