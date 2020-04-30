# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:27:39 2020

@author: PRATEEK
"""
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression

#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review = line.strip()
        rating = review[-1]
        review = review[:-1]
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels


def workingPart():
    
    rev_train,labels_train=loadData('train.txt')
#    rev_test,labels_test=loadData('test.txt')

    #Build a counter based on the training dataset
    counter = CountVectorizer(lowercase=True, stop_words=stopwords.words('english'))
    counter.fit(rev_train)


    #count the number of times each term appears in a document and transform each doc into a count vector
    counts_train = counter.transform(rev_train)#transform the training data
#    counts_test = counter.transform(rev_test)#transform the testing data #####comment this line before submission

    #train classifier
    clf = LogisticRegression(solver='liblinear', dual=False, fit_intercept=True, intercept_scaling=1,tol=0.0001, C=1.0)

    #train all classifier on the same datasets
    clf.fit(counts_train,labels_train)

    #use hard voting to predict (majority voting)
#    pred=clf.predict(counts_test) #comment this line before submission

    #print accuracy
#    print (accuracy_score(pred,labels_test)) #comment this line before submmision

if __name__=="__main__": 
        
    workingPart()



