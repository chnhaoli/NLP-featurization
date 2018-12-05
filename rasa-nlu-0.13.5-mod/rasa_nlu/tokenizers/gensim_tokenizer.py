from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Any, List

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
from gensim.utils import tokenize
if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class GensimTokenizer(Tokenizer, Component):
    name = "tokenizer_gensim"

    provides = ["token_text"]

    requires = []

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim"]
        
    @staticmethod
    def rep(text):
        import re
        regexdict = {'email_address':[r'[a-z0-9\-\_\.\<\>\:]*\@[a-z0-9\-\_\.\<\>\:]*\.[a-z0-9\-\_\.\<\>\:]*'],
                     'web_address':[r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*http[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*',
                                    r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*www3?\.[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*']}
        for key, pat_list in regexdict.items():
            for pat in pat_list:
                text = re.sub(pat, key, text)
        return text

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            text = self.rep(example.text)
            example.set("token_text", list(tokenize(text)))
            print( list(tokenize(text)))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        message.set("token_text", list(tokenize(message.text)))

