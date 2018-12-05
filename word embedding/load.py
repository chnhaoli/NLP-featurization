# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 13:52:01 2018

@author: 320028480
"""
from gensim.models import Word2Vec, KeyedVectors

gn = KeyedVectors.load_word2vec_format(r'C:\Playground\rasa_core-master\examples\agent_lisa_test\gn.bin', binary=True)
