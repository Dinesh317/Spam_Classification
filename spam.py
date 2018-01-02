import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
import string
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

################################################################
################### DATA SET READING AND #######################
################################################################

messages = pd.read_csv('/Users/dineshmaharana/jup/proj_spam_ham/smsspamcollection/SMSSpamCollection',sep = '\t',names = ["label","message"])
messages['length']= messages['message'].apply(len)
messages['length'].plot(bins = 50,kind = 'hist')
messages.hist(column='length',by ='label',bins =50,figsize = (12,6))

################################################################
################### DATA PROCESSING AND  #######################
################################################################

def text_process(mess):
    non_punc =[char for char in mess if char not in string.punctuation]
    non_punc = ''.join(non_punc)
    clean_mess = [word for word in non_punc.split() if word.lower() not in stopwords.words('english')]
    return clean_mess

################################################################
####################### CLASSIFIER #############################
################################################################

pipeline = Pipeline([('bow',CountVectorizer(analyzer = text_process)),
                     ('tfidf',TfidfTransformer()),
                     ('classifier',MultinomialNB()),
                     ])

msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size = 0.2)
pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)

################################################################
################### CLASSIFICATION REPORT ######################
################################################################

from sklearn.metrics import classification_report
print(classification_report(predictions,label_test))


from sklearn.metrics import accuracy_score
print(sklearn.metrics.accuracy_score(predictions,label_test))
