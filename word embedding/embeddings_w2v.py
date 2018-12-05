# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:16:05 2018

@author: 320028480
"""

from gensim.models import Word2Vec, KeyedVectors, FastText
from tqdm import tqdm
from math import floor
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)
def load_tokens(file="tokens.txt"):
    '''
    .txt --> list of lists of tokens
    '''
    with open(file, 'r') as f:
        sentences = f.read().split("\n")
    return [sentence.split(" ") for sentence in sentences][:-1]

lumea_tokens = load_tokens("tokens_lumea.txt")
lumea_token_count = sum([1 for sentence in lumea_tokens for token in sentence])
blog_tokens = load_tokens("tokens_blog.txt")
blog_token_count = sum([1 for sentence in blog_tokens for token in sentence])

print("Start training FT model wtih blogs.")
model = FastText(blog_tokens, 
                 size=300, 
                 window=5, 
                 min_count=1, 
                 workers=4)
model.wv.save("model_ft_blog")

print("Start updating FT model wuth Lumea corpus.")
model.build_vocab(lumea_tokens, update=True)
model.train(lumea_tokens, total_examples=model.corpus_count, epochs=model.epochs)
model.wv.save("model_ft_expanded")

print("Start training FT model wtih Lumea corpus only.")
model = FastText(lumea_tokens, 
                 size=300, 
                 window=5, 
                 min_count=1, 
                 workers=4)
model.wv.save("model_ft_lumea")

print("Start training W2V model wtih blogs.")
model = Word2Vec(blog_tokens, 
                 size=300, 
                 window=5, 
                 min_count=1, 
                 workers=4)
model.wv.save("model_w2v_blog")

print("Start updating W2V model wuth Lumea corpus.")
model.build_vocab(lumea_tokens, update=True)
model.train(lumea_tokens, total_examples=model.corpus_count, epochs=model.epochs)
model.wv.save("model_w2v_expanded")

print("Start training W2V model wtih Lumea corpus only.")
model = Word2Vec(lumea_tokens, 
                 size=300, 
                 window=5, 
                 min_count=1, 
                 workers=4)
model.wv.save("model_w2v_lumea")


'''
print("Starting visualization")

def tsne_plot(model):
    """
    Creates and TSNE model and plots it
    from https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne
    """
    labels = []
    tokens = []

    for word in tqdm(model.wv.vocab):
        if model.wv.vocab[word].count >= 65536:
            tokens.append(model[word])
            labels.append(word)
    
    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=2500)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in tqdm(new_values):
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in tqdm(range(len(x))):
zz        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
    
tsne_plot(model)
'''