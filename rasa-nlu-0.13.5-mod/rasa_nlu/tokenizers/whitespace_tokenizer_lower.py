from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
from typing import Any, List, Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.tokenizers import Tokenizer, Token
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


class WhitespaceTokenizerLower(Tokenizer, Component):
    name = "tokenizer_whitespace_lower"

    provides = ["tokens", "token_text"]
    @staticmethod
    def rep(text):
        import re
        regexdict = {'email_address':[r'[a-z0-9\-\_\.\<\>\:]*\@[a-z0-9\-\_\.\<\>\:]*\.[a-z0-9\-\_\.\<\>\:]*'],
                     'web_address':[r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*http[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*',
                                    r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*www3?\.[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*']}
        for key, pat_list in regexdict.items():
            for pat in pat_list:
                text = re.sub(pat, key, text.lower())
        return text

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.training_examples:
            tokens, token_text = self.tokenize(example.text)
            example.set("tokens", tokens)
            example.set("token_text", token_text)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        tokens, token_text = self.tokenize(message.text)
        message.set("tokens", tokens)
        message.set("token_text", token_text)

    def tokenize(self, text):
        # type: (Text) -> List[Token]

        # there is space or end of string after punctuation
        # because we do not want to replace 10.000 with 10 000
        text = self.rep(text)
        words = re.sub(r'[.,!?]+(\s|$)', ' ', text).split()

        running_offset = 0
        tokens = []
        for word in words:
            word_offset = text.index(word, running_offset)
            word_len = len(word)
            running_offset = word_offset + word_len
            tokens.append(Token(word, word_offset))
        return tokens, words
