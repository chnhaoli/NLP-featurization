from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import typing
from builtins import zip
import os
import io
from future.utils import PY3
from typing import Any, Optional
from typing import Dict
from typing import List
from typing import Text
from typing import Tuple

import numpy as np

from rasa_nlu import utils
from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message

from textblob import TextBlob as tb

logger = logging.getLogger(__name__)


class TextblobSentimentClassifier(Component):

    name = 'sentiment_classifier_textblob'

    provides = ['tb_polarity', 'tb_subjectivity']

    def __init__(self,component_config=None):
        super(TextblobSentimentClassifier, self).__init__(component_config)
        pass

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message.set("tb_polarity", self.tbPolarity(message), add_to_output=True)
        message.set("tb_subjectivity", self.tbSubjectivity(message))

    @staticmethod
    def tbPolarity(message):
        return tb(message.text).sentiment.polarity

    @staticmethod
    def tbSubjectivity(message):
        return tb(message.text).sentiment.subjectivity
