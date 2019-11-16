#!/usr/bin/env python
# coding: utf-8

# In[27]:


import nltk
from nltk.text import Text
from bs4 import BeautifulSoup
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import stopwords
import re
import pickle
import math
import numpy as np
import os

# In[103]:


def removeStopWords(wordsWithTag, language='spanish'):
    '''Receives a list of words and returns another list without stop words'''
    return [ word for word in wordsWithTag if word[0] not in stopwords.words(language) ]


# In[105]:


def clearTokens(tokensWithTag):
    '''Receives a list of  with tag and returns another list with the same tokens but only with letters'''
    result = []
    for token in tokensWithTag:
        clearToken = ""
        for c in token[0]:
            if re.match(r'[a-záéíóúñüA-ZÁÉÍÓÚÑÜ]', c):
                clearToken += c
        if len(clearToken) > 0:
            result.append((clearToken, token[1]))
    return result


# In[6]:


def getContexts(wordList, windowSize=4):
    '''Given a list of words, returns a dictionary of contexts of each word'''
    contexts = dict()
    index = 0
    for index in range(len(wordList)):
        word = wordList[index]
        if word not in contexts:
            contexts[word] = []
        start = max(0, index - windowSize)
        end = min(index + windowSize, len(wordList) - 1)
        for i in range(start, end + 1):
            if i != index:
                contexts[word].append(wordList[i])
    return contexts


# In[69]:


def getLemmasDictionary(fname="generate.txt"):
    with open(fname, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    lemmas = {}
    for line in lines:
        if line != "":
            words = [word.strip() for word in line.split()]
            wordform = words[0].replace("#", "")
            kind = words[-2][0].lower()
            lemmas[(wordform, kind)] = words[-1]
    return lemmas


# In[8]:


def serializeObject(obj, fname="lemmas2.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(obj, f, -1)


# In[9]:


def deserializeObject(fname="lemmas2.pkl"):
    with open(fname, "rb") as f:
        return pickle.load(f)


# In[126]:


def lemmatize(wordsTagged, fname="lemmas2.pkl"):
    lemmas = deserializeObject(fname)
    wordsLemmatized = []
    for word in wordsTagged:
        if word in lemmas.keys():
            wordsLemmatized.append((lemmas[word], word[1]))
        else:
            wordsLemmatized.append(word)
    return wordsLemmatized


# In[70]:

#------------------------------MAIN--------------------------------------------

serializeObject(getLemmasDictionary(), "lemmas2.pkl")


# In[19]:


def getSentences(fname):
    with open (fname, encoding="utf-8") as f:
        text_string = f.read()
    soup = BeautifulSoup(text_string, "html")
    text_string = soup.get_text()
    
    sent_tokenizer = nltk.data.load("nltk:tokenizers/punkt/english.pickle")
    sentences = sent_tokenizer.tokenize(text_string)
    return sentences


# In[147]:


sentences = getSentences("text.htm")


# In[148]:


def trainSpanishTagger():
    patterns = [
        (r'.*o$', 'n'),  # noun masculine singular
        (r'.*os$', 'n'), # noun masculine plural
        (r'.*a$', 'n'),  # noun femenine singular
        (r'.*as$', 'n')  # noun femenine plural
    ]
    regexpTagger = nltk.RegexpTagger(patterns, nltk.DefaultTagger('s'))
    unigram_tagger = nltk.UnigramTagger(nltk.corpus.cess_esp.tagged_sents(), None, regexpTagger)
    return unigram_tagger


# In[149]:


serializeObject(trainSpanishTagger(), "tagger.pkl")


# In[150]:


def tagSentencesSpanishTagger(sentences):
    sentences_tagged = []
    spanish_tagger = deserializeObject("tagger.pkl")
    for s in sentences:
        tokens = nltk.word_tokenize(s)
        s_tagged = spanish_tagger.tag(tokens)
        s_tagged = [(it[0].lower(), it[1][0].lower()) for it in s_tagged]
        sentences_tagged.append(s_tagged)
    return sentences_tagged


# In[151]:


sentencesWithTags = tagSentencesSpanishTagger(sentences)


# In[152]:


def getWordsFromSentences(sentences):
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
    return words


# In[153]:


wordsWithTags = getWordsFromSentences(sentencesWithTags)
clearedTokens = removeStopWords(clearTokens(wordsWithTags))


# In[154]:


lemmatizedWords = lemmatize(clearedTokens)
contexts = getContexts(lemmatizedWords)
vocabulary = sorted(set(lemmatizedWords))


# In[155]:


vectors = dict()
for mainWord, context in contexts.items():
    contextFreq = dict()
    for word in context:
        if word not in contextFreq:
            contextFreq[word] = 0
        contextFreq[word] += 1
    l = np.zeros(len(vocabulary))
    for i in range(len(vocabulary)):
        word = vocabulary[i]
        if word in contextFreq:
            l[i] = contextFreq[word]
    vectors[mainWord] = l


# In[156]:


ans = []
v1 = vectors[("dólar", "n")]
for word in vocabulary:
    v2 = vectors[word]
    cosine = np.dot(v1, v2) / np.sqrt(np.sum(v1 ** 2)) / np.sqrt(np.sum(v2 ** 2))
    ans.append((word, cosine))
ans.sort(key = lambda x: x[1], reverse=True)


# In[157]:


with open("output3.txt", "w") as f:
    for par in ans:
        if par[0][1] == 'n':
            f.write(F"{par[0]} {par[1]}\n")


# In[ ]:




