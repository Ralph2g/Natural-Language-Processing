#from nltk.corpus import cess_esp as brown
#brown_tagged_sents = brown.tagged_sents(categories='news')
#brown_sents = brown.sents(categories='news')
#

#tags = [tag for (word, tag) in brown.tagged_words(categories='news')]
#nltk.FreqDist(tags).max()
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

def getLemmasDictionary(fname="generate.txt"):
    with open(fname, 'r') as f:
        lines = [line.strip() for line in f.readlines()]#Obtenemos lineas
    
    lemmas = {}
    for line in lines:
        if line != "":#Mientras no sea vacía
            words = [word.strip() for word in line.split()]#Obtenemos palabras
            wordform = words[0].replace("#", "")#Removemos gato 
            kind = words[-2][0].lower()#Tipo de palabra
            lemmas[(wordform, kind)] = words[-1]#retornamos el lemma
    return lemmas

#Convertimos en un Pk los objetos
def serializeObject(obj, fname="lemmas2.pkl"):
    with open(fname, "wb") as f:
        pickle.dump(obj, f, -1)
#desconvertimos los pkl
def deserializeObject(fname="lemmas2.pkl"):
    with open(fname, "rb") as f:
        return pickle.load(f)

#Retorna las sentencias en minusculas
def get_sentences(fname):
    f=open(fname,encoding='utf-8')
    t=f.read()
    soup = BeautifulSoup(t,'lxml')
    text_string =soup.get_text() #extracción de la cadena

    #Get a list of text

    sent_tokenizer=nltk.data.load('nltk:tokenizers/punkt/spanish.pickle')
    sentences = sent_tokenizer.tokenize(text_string)
    return sentences
#Funcion de entrenamiento del tagger
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

def tagSentencesSpanishTagger(sentences):
    sentences_tagged = []
    spanish_tagger = deserializeObject("tagger.pkl")#agarramos nuestro tagger entrenado
    for s in sentences:
        tokens = nltk.word_tokenize(s)#Tokenizamos las oraicones
        s_tagged = spanish_tagger.tag(tokens)#Taggeamos los tokens
        s_tagged = [(it[0].lower(), it[1][0].lower()) for it in s_tagged]#Pasamos a minusculas
        sentences_tagged.append(s_tagged)#Agregamos las oraciones taggeadas
    return sentences_tagged
#Sacamos las palabras de las oraciones de todas las sentencias
def getWordsFromSentences(sentences):
    words = []
    for sentence in sentences:
        for word in sentence:
            words.append(word)
    return words
#Limpiamos los tokens

def removeStopWords(wordsWithTag, language='spanish'):
    '''Receives a list of words and returns another list without stop words'''
    return [ word for word in wordsWithTag if word[0] not in stopwords.words(language) ]


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

def lemmatize(wordsTagged, fname="lemmas2.pkl"):
    lemmas = deserializeObject(fname)
    wordsLemmatized = []
    for word in wordsTagged:
        if word in lemmas.keys():#Si la palabra la encuentra en el dic
            wordsLemmatized.append((lemmas[word], word[1]))#Guardamos el lemma con su tag y la agregamos a las palabras lematizadas
        else:
            wordsLemmatized.append(word)#Si no la encuentra guarda la palabra en el vocabulario en el diccionario
    return wordsLemmatized

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



def tag_sentences_regexp_tagger(sentences):
    #hacemos una tupla
    patterns=[
        (r'.*o$', 'NMS'),
        (r'.*os$', 'NMP'),
        (r'.*o$', 'NFS'),
        (r'.*as$', 'NFP')
    ]

    regexp_tagger = nltk.RegexpTagger(patterns)
    tagged = regexp_tagger.tag(sentences)
    print(tagged)


def sim(v1,v2):
    return np.sum(np.multiply(deserializeObject("idf.pkl"), np.multiply(v1,v2)))

def idf():
    vocabulary = deserializeObject("vocabulary.pkl")
    contexts = deserializeObject("contexts.pkl")
    v = []
    for word in vocabulary:
        num = 0
        for mainWord,context in contexts.items(): #Contextos de palabras ---No de frecuencias 
            if word in context:
                num +=1
        v.append(num)
    vector = np.array(v)
    serializeObject(vector,"frecuencyWordContext")
    vector = deserializeObject("frecuencyWordContext")
    idf = np.log(len(vocabulary)+1/vector)
    serializeObject(idf,"idf.pkl")
    return idf

def normalize(d1):
    result = BM25(d1)
    return result/np.sum(result)

def BM25(d1):
    k=1.5
    b=0.75
    avdl = sum( context for context in d1) / len(d1)
    BM = (( (k+1)*d1 )/( (d1) + ( k*( 1-b+(b*(len(d1))/avdl) ) ) ))
    return BM
if __name__=='__main__':

    #Tenemos que obtener el diccionario de lemmas
    #serializeObject(getLemmasDictionary(), "lemmas2.pkl")
    fname ='text.htm'
    #Obtenemos las oracioes en minusculas
    sentences = get_sentences(fname)

    #Obtenemos el Tagger entrenado
    #serializeObject(trainSpanishTagger(), "tagger.pkl")

    #Agarramos nuestras oraciones y las taggeamos
    sentencesWithTags = tagSentencesSpanishTagger(sentences)

    #Obtenemos todas las palabras con sus tags
    wordsWithTags = getWordsFromSentences(sentencesWithTags)

    #Limpiamos los Tokens y les removemos las stop words
    clearedTokens = removeStopWords(clearTokens(wordsWithTags))
    #Lematización de los tokens

    lemmatizedWords = lemmatize(clearedTokens)
    contexts = getContexts(lemmatizedWords)
    vocabulary = sorted(set(lemmatizedWords))

    vectors = dict()
    for mainWord, context in contexts.items():
        contextFreq = dict()
        for word in context:
            if word not in contextFreq:
                contextFreq[word] = 0
            contextFreq[word] += 1#Si la palabra está en el contexto le pone un uno
        l = np.zeros(len(vocabulary))#Vector tam vocabulary de 0's
        for i in range(len(vocabulary)):
            word = vocabulary[i]#Agarramos la palabra del vocabulario
            if word in contextFreq:#Si la palabra esta en la frecuancia de contexto
                l[i] = contextFreq[word]#Ponle un sumale la frecuencia a la palabra 
        vectors[mainWord] = l
        
    #serializeObject(contexts,"contexts.pkl")
    #serializeObject(vocabulary, "vocabulary.pkl")
    ans = []
    ans2 = []
    w = ("dólar", "n")
    v1 = vectors[w]
    for word in vocabulary:
        v2 = vectors[word]
        #Agregamos el metodo  que compara IDF y BM25
        vec1 = normalize(v1)
        vec2 = normalize(v2)
        IDF = sim(vec1,vec2)

        idf1 = (deserializeObject("idf.pkl")*vec1)
        idf2 = (deserializeObject("idf.pkl")*vec2)

        cosine = np.dot(v1, v2) / np.sqrt(np.sum(v1 ** 2)) / np.sqrt(np.sum(v2 ** 2)) 
        cosine2 = np.dot(idf1, idf2) / np.sqrt(np.sum(idf1 ** 2)) / np.sqrt(np.sum(idf2 ** 2))
        ans.append((word, IDF ,cosine))
        ans2.append( ( word, cosine2 ) )
    ans.sort(key = lambda x: x[1], reverse=True)
    ans2.sort(key = lambda x: x[1], reverse=True)

    
    with open("output5.txt", "w") as f:
        for par in ans:
            if par[0][1] == 'n':
                f.write(F"{par[0]} {par[1]}\n")

    with open("output2.txt", "w") as f:
        for par in ans2:
            if par[0][1] == 'n':
                f.write(F"{par[0]} {par[1]}\n")
