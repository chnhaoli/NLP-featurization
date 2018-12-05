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


class ArbitaryRestart(Component):

    name = "arbitary_restart"
    
    provides = ["intent"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        if message.text == '-restart':
            message.set("intent", {"name":"restart", "confidence": 1.0})
        print('\x1b[6;30;42m' + str(message.get("intent", "Not able to get intent.")) + '\x1b[0m') 
