from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import map
from typing import Any
from typing import Dict
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message


class ArbitaryConfidenceRetriever(Component):

    name = "arbitary_confidence_retriever"
    
    provides = ["arbitary_confidence"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        if message.text[-1] == '%':
            message.set("arbitary_confidence", float(message.text[-3:-1])/100)
            message.text = message.text[:-3].strip()
