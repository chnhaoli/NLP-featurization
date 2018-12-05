# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 17:16:05 2018

@author: 320028480
"""

import pandas as pd
from tqdm import tqdm
from math import floor
import re
def rep(corpus):
    regexdict = {'email_address':[r'[a-z0-9\-\_\.\<\>\:]*\@[a-z0-9\-\_\.\<\>\:]*\.[a-z0-9\-\_\.\<\>\:]*'],
             'web_address':[r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*http[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*',
                            r'[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*www3?\.[a-z0-9\\\/\:\.\?\#\=\@\>\<\&\;\%\-\_\[\]\+]*']}
    for key, pat_list in regexdict.items():
        for pat in pat_list:
            corpus = re.sub(pat, key, corpus)
    return corpus
    

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

print("Start Preprocessing.")

# if not token.is_punct 
# Container of full training text, list of sentences in tokens
full = ''

txt_files = ['Brand Activation.txt',
             'Call centre training.txt', 
             'Lumea IPL intro.txt', 
             'PIM Booklet.txt',
             'Retail training.txt',
             'User Manual.txt',
             'Lumea Webpage.txt']
for file in txt_files:
    full = full + '\n' + clean(file)

with open('data_ft_lumea.txt', 'w', encoding='utf-8') as f:
    f.write(full)

with open('data_ft_expanded.txt', 'w', encoding='utf-8') as f:
    f.write(full)
    
# Saleforce
sf = pd.read_excel('entries.xlsx').drop_duplicates().astype(str)['Descr_Org']
sf = list(sf)
full = full + ''.join(sf)

print("Start training W2V model wtih blogs.")
file = 'blogtext.csv'
with open(file, 'r', encoding='utf-8') as f:
    length = sum([1 for line in f])
batch_size = 4096
blogtext = ''
for idx in tqdm(range(floor(length/batch_size))):
    batch = pd.read_csv(file, usecols=[6], header=None, skiprows=batch_size*idx, nrows=batch_size).drop_duplicates().astype(str)[6].str.strip()
    batch_corpus = '\n'.join(list(batch)) + '\n'
    blogtext += batch_corpus
    
with open('data_ft_expanded.txt', 'a', encoding='utf-8') as f:
    f.write(blogtext)

with open('data_ft_blogtext.txt', 'w', encoding='utf-8') as f:
    f.write(blogtext)
