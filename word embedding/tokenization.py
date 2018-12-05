#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:14:35 2018

@author: hermannlee
"""
import spacy
import re
import pandas as pd
from tqdm import tqdm

LOWER = False

try :
    if not isinstance(nlp, spacy.lang.en.English):
        nlp = spacy.load('en_core_web_lg')
except:
    nlp = spacy.load('en_core_web_lg')
def tokenize(sentence, lower):
    if lower:
        sentence = sentence.lower()
    regexdict = {'email_address':[r'[a-z0-9\-\_\.\<\>\:]*\@[a-z0-9\-\_\.\<\>\:]*\.[a-z0-9\-\_\.\<\>\:]*'],
             'web_address':[r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*http[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*',
                            r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*www3?\.[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*']}
    for key, pat_list in regexdict.items():
        for pat in pat_list:
            sentence = re.sub(pat, key, sentence)
    return [token.text for token in nlp(sentence)]
'''

def tokenize(sentence, lower=False):
    # type: (Text) -> List[Token]

    # there is space or end of string after punctuation
    # because we do not want to replace 10.000 with 10 000
    if lower:
        sentence = sentence.lower()

    regexdict = {'email_address':[r'[a-z0-9\-\_\.\<\>\:]*\@[a-z0-9\-\_\.\<\>\:]*\.[a-z0-9\-\_\.\<\>\:]*'],
                 'web_address':[r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*http[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*',
                                r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*www3?\.[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*']}
    for key, pat_list in regexdict.items():
        for pat in pat_list:
            sentence = re.sub(pat, key, sentence)
    return re.sub(r'[.,!?]+(\s|$)', ' ', sentence).split()
'''

print("Start Preprocessing.")

def clean(file):
    try:
        with open(file, 'r') as f:
            text = f.read()
    except:
        with open(file, 'r', encoding="utf-8") as f:
            text = f.read()
        
 
    # Remove extra newlines.
    if file == 'User Manual.txt':
        text = text.split('\n')
        for line in text:
            line = line.strip('1234567890')
        text = '\n'.join(text)
    return text

def preprocess_txt(file):
    print("Processing " + file)
    text = clean(file)
    tokenized_sentences = [tokenize(sentence, LOWER) for sentence in text.split('\n')]
    print("Done!")
    return tokenized_sentences

def save_tokens(tokens, file="tokens.txt"):
    '''
    list of lists of tokens -> .txt
    '''
    print("Saving to " + file)
    with open(file, 'w') as f:
        f.write('')
    for sentence in tqdm(tokens):
        with open(file, 'a') as f:
            f.write(" ".join(sentence) + "\n")
    print("Successfully saved to " + file + "!")
# if not token.is_punct 
# Container of full training text, list of sentences in tokens
full = [] 

txt_files = ['Brand Activation.txt',
             'Call centre training.txt', 
             'Lumea IPL intro.txt', 
             'PIM Booklet.txt',
             'Retail training.txt',
             'User Manual.txt',
             'Lumea Webpage.txt']
for file in txt_files:
    full = full + preprocess_txt(file)

# Saleforce
print("Processing Salesforce data")
sf = pd.read_excel('entries.xlsx').drop_duplicates().astype(str)['Descr_Org']
sf = [tokenize(sentence, LOWER) for sentence in tqdm(sf)]
full = full + sf
print("Done!")
save_tokens(full, file="tokens_lumea.txt")

print("Processing blog corpus")
file = 'blogtext.csv'
with open("tokens_blog.txt", 'w') as f:
    f.write('')
blogs = pd.read_csv(file, usecols=[6], header=None).drop_duplicates().astype(str)[6].str.strip()
#blog = [tokenize(sentence, LOWER) for sentence in tqdm(blog)]
with open("tokens_blog.txt", 'a') as f:
    for blog in tqdm(blogs):    
        #f.write(str(n)+'\n')
        f.write(' '.join(tokenize(blog, LOWER)) + '\n')    
    
print("Done!")
#save_tokens(blog, file="Blog_tokens.txt")



