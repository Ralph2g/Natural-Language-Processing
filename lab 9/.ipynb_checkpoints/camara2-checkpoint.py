#Creamos los vectores X y Y definidos en clase
def create_vector_x_y():
    f=open('SMS_Spam_Corpus_big.txt')
    lines=f.readlines()
    #print(lines)
    #print(len(lines))
    X_corpus = [line.lower() for line in lines]
    X_corpus = [nltk.word_tokenize(line) for line in X_corpus]
    X_texts = [line[:-2] for line in X_corpus]
    #print("The size of x_texts is %d"%len(x_texts))

    vec_y=[]
    for text in X_corpus:
        tag = text[-1].strip()
        if tag=="spam":
            vec_y.append(1)
        else:
            vec_y.append(0)
    #Vamos a taggear
    X_pos_tagged  = []
    for text in X_corpus:
        pos_tagged_text = nltk.pos_tag(text)
        X_pos_tagged.append(pos_tagged_text)
    
    #print("\nText Post tagged \n")
    #print(X_pos_tagged[500])
    
    #Vamos a lematizar
    from nltk import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    X_lemmatized = []
    for text in X_pos_tagged:
        lemmas = []
        for token in text:
            str1 = token[0]
            str2 = token[1].lower()
            try:
                str2 = str2[0]
                lemma = lemmatizer.lematize(str1,str2)
                lemmas.append(lemma.lower())
            except:
                lemmas.append(str1.lower())
        lemmas_string=' '.join(lemmas)
        X_lemmatized.append(lemmas_string)
    return X_lemmatized,vec_y
                

def classify_sklearn(X,y):
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect=CountVectorizer()
    X_counts=count_vect.fit_transform(X)
    
    classifiers=[]
    
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf_transformer=TfidfTransformer()
    X_tfidf=tfidf_transformer.fit_transform(X_counts)
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X_tfidf,y,test_size=0.2,random_state=42)
    
    from sklearn.naive_bayes import MultinomialNB
    clf_1=MultinomialNB()
    classifiers.append((clf_1,'MultinomialNB'))
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix
    clf=classifiers[0][0]
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    print(classifiers[0][1])
    print('Accuracy of prediction in', clf.score(X_test,y_test))
    print('Confusion matrix:\n',confusion_matrix(y_test,y_pred))
    print(metrics.classification_report(y_test,y_pred))

               
if __name__=='__main__':
    X,y=create_vector_x_y()
    classify_sklearn(X,y)
    
    
    
    
    
    