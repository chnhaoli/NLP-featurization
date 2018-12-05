from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core import utils
from rasa_core_sdk import Action
from rasa_core.agent import Agent
from rasa_core_sdk.events import SlotSet
from rasa_core.featurizers import (
    MaxHistoryTrackerFeaturizer,
    BinarySingleStateFeaturizer)
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies.fallback import FallbackPolicy
from rasa_core.policies.confirmation import ConfirmationPolicy

from rasa_core import run

import pprint
pp = pprint.PrettyPrinter(indent=4)

# Disable AVX warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

logger = logging.getLogger(__name__)

POLICIES = [ConfirmationPolicy(),
            KerasPolicy(),
            MemoizationPolicy()]

def train_dialogue(domain_file="domain_lisa.yml",
                   model_path="models/current/dialogue",
                   training_data_file="data/stories"):
    agent = Agent(domain_file,
                  POLICIES)

    training_data = agent.load_data(training_data_file,
                                    augmentation_factor=0)
    agent.train(
            training_data,
            epochs=400,
            batch_size=100,
            validation_split=0.2
    )

    agent.persist(model_path)
    return agent


def train_nlu():
    from rasa_nlu.training_data import load_data
    from rasa_nlu import config
    from rasa_nlu.model import Trainer

    training_data = load_data('data/nlu/')
    trainer = Trainer(config.load("nlu_model_demo.yml"))
    trainer.train(training_data)
    model_directory = trainer.persist('models',
                                      fixed_model_name="nlu",
                                      project_name="current")

    return model_directory


def run_cmd(core="models/current/dialogue", nlu="models/current/nlu", endpoint="endpoints.yml"):
    from rasa_core import run 
    _endpoints = run.AvailableEndpoints.read_endpoints(endpoint)
    _interpreter = run.NaturalLanguageInterpreter.create(nlu,
                                                         _endpoints.nlu)
    _agent = run.load_agent(core_model=core,
                        interpreter=_interpreter,
                        endpoints=_endpoints)

    run.serve_application(_agent)

def serve_actions(actions="actions"):
    from rasa_core_sdk import endpoint as ep
    edp_app = ep.endpoint_app(cors_origins=None,
                           action_package_name=actions)
    http_server = ep.WSGIServer(('0.0.0.0', ep.DEFAULT_SERVER_PORT), edp_app)

    http_server.start()
    http_server.serve_forever()

def run_online(interpreter="models/current/nlu",
               domain_file="domain_lisa.yml",
               training_data='data/stories',
               out="models/current/dialogue",
               endpoints="endpoints.yml"):
    from rasa_core import train
    _endpoints = train.AvailableEndpoints.read_endpoints(endpoints)
    _interpreter = train.NaturalLanguageInterpreter.create(interpreter,
                                                     _endpoints.nlu)
    _agent = Agent(domain_file,
                  generator=_endpoints.nlg,
                  action_endpoint=_endpoints.action,
                  interpreter=_interpreter,
                  policies=POLICIES)
    training_data = _agent.load_data(training_data)
    _agent.train(training_data)
    _agent.persist(out)
    train.online.run_online_learning(_agent, finetune=False)



if __name__ == '__main__':
    utils.configure_colored_logging(loglevel="INFO")

    parser = argparse.ArgumentParser(
            description='starts the bot')

    parser.add_argument(
            'task',
            choices=["train-nlu", "train-dialogue", "run", "train-online", "serve-actions"],
            help="what the bot should do - e.g. run or train?")
    task = parser.parse_args().task

    # decide what to do based on first parameter of the script
    if task == "train-nlu":
        train_nlu()
    elif task == "train-dialogue":
        train_dialogue()
    elif task == "run":
        run_cmd()
    elif task == "train-online":
        run_online()
    elif task == "serve-actions":
        serve_actions()
