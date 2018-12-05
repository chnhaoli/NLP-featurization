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


class ArbitaryConfidenceSetter(Component):

    name = "arbitary_confidence_setter"

    requires = ["arbitary_confidence", "intent"]
    
    provides = ["intent"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        if message.get("arbitary_confidence"):
            intent = message.get("intent")
            intent["confidence"] = message.get("arbitary_confidence")
            message.set("intent", intent,
                        add_to_output=True)
