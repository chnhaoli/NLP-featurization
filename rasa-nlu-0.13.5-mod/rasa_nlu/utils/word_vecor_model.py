from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from rasa_nlu.model import Metadata


class WordVectorModel(Component):
    name = "word_vector_model"

    provides = ["word_vectors", "wv_model"]

    requires = ["token_text"]

    defaults = {
        # name of the language model to load - if it is not set
        # we will be looking for a language model that is named
        # after the language of the model, e.g. `en`
        "model": "model_expanded_lower.bin"
    }
    def __init__(self, component_config=None, model=None):
        # type: (Dict[Text, Any], Language) -> None
        self.model = model
        super(WordVectorModel, self).__init__(component_config)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim", "numpy"]

    @classmethod
    def create(cls, cfg):
        # type: (RasaNLUModelConfig) -> GoogleNewsWV
        component_conf = cfg.for_component(cls.name, cls.defaults)
        wv_model_name = component_conf.get("model")

        # if no model is specified, we fall back to the language string
        if not wv_model_name:
            wv_model_name = 'model_expanded.bin'
            component_conf["model"] = 'model_expanded.bin'

        logger.info("Trying to load word2vec model with "
                    "name '{}'".format(wv_model_name))

        try:
            model = KeyedVectors.load(wv_model_name)
        except:
            try:
                model = KeyedVectors.load_word2vec_format(wv_model_name,binary=True)
            except:
                raise FileNotFoundError('No trained W2V model given.')

        return WordVectorModel(component_conf, model)

    @classmethod
    def cache_key(cls, model_metadata):
        # type: (Metadata) -> Text

        component_meta = model_metadata.for_component(cls.name)

        # Fallback, use the language name, e.g. "en",
        # as the model name if no explicit name is defined
        spacy_model_name = component_meta.get("model", model_metadata.language)

        return cls.name + "-" + spacy_model_name

    def provide_context(self):
        # type: () -> Dict[Text, Any]

        return {"wv_model": self.model}

    def wv_for_tokens(self, sentence):
        wvs = []
        #v = 0
        #oov = 0
        for word in sentence:
            try:
                wv = self.model[word]
                #v += 1
            except:
                wv = np.zeros(self.model.wv.vector_size)
                #print(word)
                #oov += 1
            wvs.append(wv)
        #print("Having {} vocab words and {} oov words".format(v, oov))
        return np.array(wvs).T

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            example.set("word_vectors", self.wv_for_tokens(example.get("token_text")))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("word_vectors", self.wv_for_tokens(message.get("token_text")))

    @classmethod
    def load(cls,
             model_dir=None,
             model_metadata=None,
             cached_component=None,
             **kwargs):
        # type: (Text, Metadata, Optional[SpacyNLP], **Any) -> SpacyNLP

        if cached_component:
            return cached_component

        component_meta = model_metadata.for_component(cls.name)
        model_name = component_meta.get("model")

        try:
            model = KeyedVectors.load(wv_model_name)
        except:
            try:
                model = KeyedVectors.load_word2vec_format(wv_model_name,binary=True)
            except:
                raise FileNotFoundError('No trained W2V model given.')

        return cls(component_meta, model)
