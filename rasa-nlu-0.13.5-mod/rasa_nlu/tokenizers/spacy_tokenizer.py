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
import numpy as np
if typing.TYPE_CHECKING:
    from spacy.tokens.doc import Doc


class SpacyTokenizer(Tokenizer, Component):
    name = "tokenizer_spacy"

    provides = ["tokens", "token_text", "word_vectors"]

    requires = ["spacy_doc"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            tokens, token_text = self.tokenize(example.get("spacy_doc"))
            example.set("tokens", tokens)
            example.set("token_text", token_text)
            example.set("word_vectors", self.wv_for_doc(example.get("spacy_doc")))
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

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        tokens, token_text = self.tokenize(message.get("spacy_doc"))
        message.set("tokens", tokens)
        message.set("token_text", token_text)
        message.set("word_vectors", self.wv_for_doc(message.get("spacy_doc")))
    def tokenize(self, doc):
        # type: (Doc) -> List[Token]

        return [Token(t.text, t.idx) for t in doc], [self.rep(t.text) for t in doc if (not t.is_punct)]

    def wv_for_doc(self, doc):
        return np.array([t.vector for t in doc if (not t.is_punct)]).T

# if (not t.is_punct) and (t.text not in ["a","and","of","to"])