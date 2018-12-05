# Agent Test Version 

## Environment Setup: 
This project has only been ran on a Windows 10 machine. It's recommended to work in a virtual environment using `vitrualenv` or `conda` while it's not required, thus won't be instructed here.

### Backend Engine
1. Install Python 3.6.6. When developing, this project utilized the Python distribution with Anaconda 1.8.7.
2. Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for package building.
3. Clone / download this repository to you local machine and set it as working directory.
4. `pip install -e rasa-nlu-0.13.5-mod` installs modified `rasa_nlu` package. If you are experiencing problems with this, try to run `pip install incremental` and `pip install Twisted` first.
5. `python -m spacy download en_core_web_md` or `python -m spacy download en_core_web_lg` to download spaCy language model. `-lg` model for better performance while `-md` model for less memory usage.
6. `python -m spacy link en_core_web_md en` or `python -m spacy link en_core_web_lg en` correspondingly in order to use `en` as link to model. 
7. If using some word embeddings other than SpaCy model, place the file in current directory, and change the NLU pipeline accordingly.
8. `pip install -e rasa-core-0.11.8-mod` installs modified `rasa_core` package.
9. `pip install -e rasa-core-sdk-0.11.5-mod` installs modified `rasa_core_sdk` package.
10. For using fastText classifier, install python fastText as per https://github.com/facebookresearch/fastText/tree/master/python


### Frontend Build
1. Install [yarn](https://yarnpkg.com/lang/en/docs/install), tested with 1.12.1.
2. Set `UI` as your working directory. If you are now in project root directory just do it by `cd UI`.
3. `yarn install` to install frontend dependencies. If you are experiencing problems with this, try to install a python 2.x distribution first, but please make sure all other scripts ran in python 3.x as they are only tested in python 3.x.
4. `yarn build` to build frontend files. 

## Usage
### Backend
In your command-line interface (CLI) set working directory as this very directory

Usage | Command | Make | Comment
---|---|---|---
Train NLU | `python bot.py train-nlu` | | 
Train dialogue model | `python bot.py train-dialogue` | `make train-core` | 
Serve actions | `python bot.py serve-actions` | `make action-server` | Mandatory to run in a separatory CLI before running chatbot if using custom actions.  
Interactive training in CLI | `python bot.py train-online` |  | 
Run chatbot in CLI | `python bot.py run` | `make cmdline` | Need trained NLU and dialogue models

### Frontend
1. Make sure you have *Environment Setup* completed, and have NLU model and dialogue model trained;
2. In one CLI window, set the project root directory as working directory, run `python bot.py serve-actions` to host a server for custom actions;
3. In another CLI window, set the project root directory as working directory, run `python -m bot_demo -d models/current/dialogue -u models/current/nlu --endpoint endpoints.yml` to host a server for the chatbot;
4. (Optional) use _ngrok_ to expose your local host to public:
   1. Download [ngrok](https://ngrok.com/) into working directory
   2. In another CLI window, `ngrok http 5005` to expose the backend server.
   3. In `UI/index.html`, change the host as stated in previous `ngrok` step.
   4. In another CLI window, `ngrok http 8080` to expose the frontend server. 
5. In another CLI window, go to `UI` folder, run `yarn serve` to host the frontend webpage;
6. In your web browser, go to the address indicated in step 4.3 or 5.

### Notes
- To visualize the training data please go to https://rasahq.github.io/rasa-nlu-trainer/ and upload the corresponding data file via 'Click to Upload' at the upper-right corner.

## Repository Overview

Repository / File | Description
---: | :---
`data/nlu/` | NLU data. Real data are not uploaded.
`data/stories/` | Dialogue data.
`rasa-core-0.11.8-mod` | Modified `rasa-core` module.
`rasa-core-sdk-0.11.5-mod` | Modified `rasa-core-sdk` module.
`rasa-nlu-0.13.5-mod` | Modified `rasa-nlu` module.
`UI` | A simple webpage demo.
`__init__.py` | Versions.
`.gitignore` | .gitignore file.
`actions.py`| Scripts of custom actions.
`bot_demo.py` | Scripts for running chatbot for frontend.
`bot_server_channel.py` | Scripts for communication channel between chatbot and frontend.
`bot.py` | Scripts for commandline usage.
`domain_lisa.yml` | Chatbot domain, where defines intents, slots, actions and utter templates in use.
`endpoints.yml`| Endpoints for actions HTTP server.
`Makefile` | Makefile for `make` scripts in UNIX like systems. Only for reference purposes and not tested.
`message_store.json` | Logged interaction from frontend. 
`nlu_model_config.yml` | NLU pipeline configuration for testing.
`nlu_model_demo.yml` | NLU pipeline for the demo chatbot.


## Evaluation
### NLU

Evaluation (Not for Gitlab Version)
```python -m rasa_nlu.evaluate --data data/nlu_split/test --model models/current/nlu```

Cross-validation
```python -m rasa_nlu.evaluate --data data/data_world_no_ner.md --mode crossvalidation --config nlu_model_config.yml --folds 5```

### Dialogue

```python -m rasa_core.evaluate -d models/current/dialogue -s data/stories -o matrix.pdf --failed failed_stories.md```

## Confirmation fallback notes

 1. `ConfirmationPolicy()` must be put at the first place when declaring policy usage.
 2. Default `nlu_threshold=0.1`.
 3. Can now add "nn%" at the end of input to force NLU to set intent confidence as nn%. e.g. `it does not work 02%`