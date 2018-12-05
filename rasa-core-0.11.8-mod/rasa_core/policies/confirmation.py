from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import json
import io
import typing
import copy

from typing import Any, List, Text

from rasa_core import utils
from rasa_core.policies.policy import Policy
from rasa_core.constants import FALLBACK_SCORE
from rasa_core.events import (
    UserUttered, ActionExecuted,
    Event, SlotSet, Restarted, ActionReverted, UserUtteranceReverted,
    BotUttered)


logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker


class ConfirmationPolicy(Policy):
    """Policy which executes a confirmation action if NLU confidence is low
       Should be put before other prediction policies.
        :param float nlu_threshold:
          minimum threshold for NLU confidence.
          If intent prediction confidence is lower than this,
          predict confirmation action with confidence 1.0.

        :param Text confirmation_action_name:
          name of the action to execute as a confirmation.
    """

    @staticmethod
    def _standard_featurizer():
        return None

    def __init__(self,
                 nlu_threshold=0.1,  # type: float
                 confirmation_action_name="action_default_confirmation", # type: Text
                 confirmation_rephrase_name="action_default_rephrase",  # type: Text
                 affirm_intent_name="affirm",  # type: Text
                 deny_intent_name="deny"  # type: Text
                 ):
        # type: (...) -> None

        super(ConfirmationPolicy, self).__init__()

        self.nlu_threshold = nlu_threshold
        self.confirmation_rephrase_name = confirmation_rephrase_name
        self.confirmation_action_name = confirmation_action_name
        self.affirm_intent_name = affirm_intent_name
        self.deny_intent_name = deny_intent_name


        self.cache_message = UserUttered(None)
    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: Any
              ):
        # type: (...) -> None
        """Does nothing. This policy is deterministic."""

        pass

    def should_confirm(self,
                        nlu_confidence,  # type float
                        last_action_name  # type: Text
                        ):
        # type: (...) -> bool
        """It should predict confirmation action only if
        a. predicted NLU confidence is lower than ``nlu_threshold`` &&
        b. last action is NOT confirmation action
        """
        return (nlu_confidence < self.nlu_threshold and
                last_action_name != self.confirmation_action_name)

    def confirmation_scores(self, domain, confirmation_score=FALLBACK_SCORE):
        """Prediction scores used if a confirmation is necessary."""

        result = [0.0] * domain.num_actions
        idx = domain.index_for_action(self.confirmation_action_name)
        result[idx] = confirmation_score
        return result

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts a confirmation action if NLU confidence is low
        """

        nlu_data = tracker.latest_message.parse_data

        # if NLU interpreter does not provide confidence score,
        # it is set to 1.0 here in order
        # to not override standard behaviour
        nlu_confidence = nlu_data["intent"].get("confidence", 1.0)
        '''
        print("\033[92m Events\033[0m")
        for idx, event in enumerate(tracker.events):
            print(str(idx) + '. ' + str(event))
        print("\033[92m Applied Events\033[0m")
        for idx, event in enumerate(tracker.applied_events()):
            print(str(idx) + '. ' + str(event))
        print("\033[33mLatest action:{}\033[0m".format(tracker.latest_action_name))
        print("\033[33mLatest intent:{}\033[0m".format(nlu_data["intent"]["name"]))
        '''
        # know what bot just did before 'action_listen'
        action_history = [evt.action_name for evt in tracker.events if isinstance(evt, ActionExecuted)]
        proceeding_action = action_history[-2] if len(action_history) > 2 else 'action_listen'

        if proceeding_action == self.confirmation_action_name:
            # Last action is confirmation question
            if nlu_data["intent"]["name"] == self.affirm_intent_name:
                # TODO: The user affirms the intent in confirmation, thus,
                # predict next action based on previous intent 
                # Revert current `confirm (action) <- affirm (intent)`
                # And retrieve cached message for other policy to predict

                from rasa_core.events import UserUtteranceReverted
                tracker.update(UserUtteranceReverted())
                tracker.update(ActionExecuted('action_listen'))
                tracker.update(self.cache_message)
                tracker.latest_message = self.cache_message
                result = [0.0] * domain.num_actions


            elif nlu_data["intent"]["name"] == self.deny_intent_name:
                # TODO: The user denies the intent in confirmation, thus,
                # predict next action as ask for rephrasing 
                # Revert current `confirm (action) <- deny (intent)`.
                from rasa_core.events import UserUtteranceReverted
                result = [0.0] * domain.num_actions
                idx = domain.index_for_action(self.confirmation_rephrase_name)
                result[idx] = FALLBACK_SCORE
                
            else:
                # TODO: The user gives a rephrase, thus,
                # predict next action based on current intent 
                # Just return all 0 and let prediction policy do.
                result = [0.0] * domain.num_actions

        elif proceeding_action == self.confirmation_rephrase_name:
            # TODO: The user shall give another intent after bot ask for rephrase.
            # So predict based on new intent.
            result = [0.0] * domain.num_actions
            
        elif self.should_confirm(nlu_confidence if nlu_confidence else 1, tracker.latest_action_name):
            logger.debug("NLU confidence {} is lower "
                         "than NLU threshold {}. "
                         "Predicting confirmation action: {}"
                         "".format(nlu_confidence, self.nlu_threshold,
                                   self.confirmation_action_name))
            # we set this to 1.1 to make sure confirmation overrides
            # the memoization policy
            # Meanwhile we need to cache the current message for later prediction,
            # then revert input intent and confirmation action.
            self.cache_message = copy.deepcopy(tracker.latest_message)
            self.cache_message.parse_data['intent']['confidence'] = 1.0
            result = self.confirmation_scores(domain)
        elif tracker.latest_action_name == self.confirmation_action_name:
            result = [0.0] * domain.num_actions
            idx = domain.index_for_action('action_listen')
            result[idx] = FALLBACK_SCORE
        else:
            # NLU confidence threshold is met, as well as not in a confirmation process,
            # so predict all 0, let other policies predict.
            result = [0.0] * domain.num_actions
        '''
        print("\033[92m Events\033[0m")
        for idx, event in enumerate(tracker.events):
            print(str(idx) + '. ' + str(event))
        print("\033[92m Applied Events\033[0m")
        for idx, event in enumerate(tracker.applied_events()):
            print(str(idx) + '. ' + str(event))
        '''
        return result

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to storage."""
        config_file = os.path.join(path, 'confirmation_policy.json')
        meta = {
            "nlu_threshold": self.nlu_threshold,
            "confirmation_action_name": self.confirmation_action_name
        }
        utils.create_dir_for_file(config_file)
        utils.dump_obj_as_json_to_file(config_file, meta)

    @classmethod
    def load(cls, path):
        # type: (Text) -> ConfirmationPolicy
        meta = {}
        if os.path.exists(path):
            meta_path = os.path.join(path, "confirmation_policy.json")
            if os.path.isfile(meta_path):
                with io.open(meta_path) as f:
                    meta = json.loads(f.read())

        return cls(**meta)
