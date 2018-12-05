# Rasa NLU Development Guide

This is modified on top of version 0.13.5 of `rasa_nlu`. Originally docs of `rasa_nlu` please refer to https://rasa.com/docs/nlu/.

## Pipeline Customization Procedure
1. Inspect `components.py --> class Component(object)` to get an idea how a pipeline component looks like;
2. Create a python file for the component, e.g., `sentiment/textblob_sentiment_classifier.py`;
 - `name` for pipeline configure file;
 - `require` and `provide` data from / for other pipeline components;
 - In `process`, provide output data by `message.set('provide_data')` according what stated in `provide`;
3. Register the component in `registry.py`.

## Exposing Data to Rasa Core
- Look into `model.py`;
- In class `Interpreter` defined method `parse`, who handles the pipeline processing on text;
- Method `parse` returns a dictionary contains both default properties and message information

## Major Changes against Original Rasa NLU
1. `registry.py`
   - Registration for customized components. 
    * Import the component class at the beginning;
    * Register the class name in `component_classes`.
2. `./classifiers/embedding_intent_classifier.py`
   - Allow manipulation of learning rate in configuration file.
3. `./classifiers/fasttext_classifier.py`
   - [Facebook fastText classifier](https://fasttext.cc/docs/en/references.html).
4. `./featurizers/embedding_average_featurizer.py`
   - Sentence embedding by averaging all token word embeddings.
5. `./featurizers/embedding_minmax_featurizer.py`
   - Sentence embedding by concatenating minimums and maximums of all token word embeddings.
6. `./featurizers/embedding_tfidf_featurizer.py`
   - Sentence embedding by averaging all token word embeddings weighted by TF-IDF.
7. `./sentanalyzers/textblob_sentiment_classifier.py`
   - Sentiment classification using `TextBlob`;
8. `./tokenizers/gensim_tokenizer.py`
   - Sentence tokenizer migrated from `gensim` module.
9. `./tokenizers/spacy_tokenizer.py`
   - Now provides word vectors of non-punctuation tokens
   - Now provides texts for non-punctuation tokens
   - Replace weblink and email address with respective tokens
10. `./tokenizers/whitespace_tokenizer.py`
   - Now provides texts of tokens
   - Replace weblink and email address with respective tokens
11. `./tokenizers/whitespace_tokenizer_lower.py`
   - Lower text version for `whitespace_tokenizer.py`
12. `./utils/arbitary_confidence_retriever.py`
   - Get an arbitary confidence score from message (xx% at the end) and trim it.
13. `./utils/arbitary_confidence_setter.py`
   - Set the confidence score to the arbitary confidence.
14. `./utils/arbitary_restart.py`
   - Give an arbitary intent to text input `-restart`.
15. `./utils/tfidf_calculator.py`
   - Calculate the TF-IDF of each token.
16. `./utils/word_vecor_model.py`
   - Retrieve embedding for each token word from pre-trained `Word2Vec` model.

## Pipeline Configs
1. Pay attention to what each component `requires` and `provides`. All component `requires` must be satisfied before the component instanciated.
2. At least your pipeline should finally provide `intent`.
3. If needed, `arbitary_confidence_setter` and `arbitary_restart` should be placed after the major intent classifier in the pipeline.
4. If needed, pre-trained word vectors should be placed in the agent working directory (not this directory), and modify the agent pipeline configuration file accordingly.
5. When using multiple sentence featurizers, previous feature vectors will be elongated, not replaced. 