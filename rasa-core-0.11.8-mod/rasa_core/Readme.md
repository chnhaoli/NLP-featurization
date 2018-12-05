# Rasa Core Source Playground
This is modified on top of version 0.11.8 of `rasa_core`. Originally docs of `rasa_core` please refer to https://rasa.com/docs/core/.

## Acquiring Data from Rasa NLU
- In Rasa Core `processor.py`, `_parse_message()` parses message using Rasa NLU interpreter `.parse` method;
- NLU interpreter returns the parsed data in the form of a *dictionary*;
- The parsed data will be updated to the ***tracker***, as an instance of `UserUttered` *class* (child of `Event`), appended in a list of events;
- In an instance of `UserUttered` class, the parsed data can be accessed using `.parse_data`.

## Changes against Original Rasa Core:
1. `processor.py`
 - In `handle_message()` added `pprint` for testing purpose.
2. `trackers.py`
 - In `DialogueStateTracker.current_state()` added `sentiments` and `latest_sentiment` for public interface.
3. `agent.py`
 - In `load_data()` changed default parameter from `augmentation_factor=20` to `augmentation_factor=0`.
4. `confirmation.py`
 - New. Implement a policy execute a confirmation question when NLU confidence is low and predict later actions based on answers to confirmation questions.
