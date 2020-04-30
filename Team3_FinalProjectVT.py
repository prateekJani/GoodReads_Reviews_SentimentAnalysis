# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 20:27:39 2020

@author: PRATEEK
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


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
	rev_test,labels_test=loadData('test.txt')

	#Build a counter based on the training dataset
	counter = CountVectorizer(lowercase=True, stop_words=stopwords.words('english'))
	counter.fit(rev_train)


	#count the number of times each term appears in a document and transform each doc into a count vector
	counts_train = counter.transform(rev_train)#transform the training data
	counts_test = counter.transform(rev_test)#transform the testing data

	#train classifier
	model1 = DecisionTreeClassifier()
	model2 = MultinomialNB()
	model3=LogisticRegression(solver='liblinear')

	predictors=[('nb',model1),('dt',model2),('lreg',model3)]

	VT=VotingClassifier(predictors)

	#train all classifier on the same datasets
	VT.fit(counts_train,labels_train)

	#use hard voting to predict (majority voting)
	pred=VT.predict(counts_test)

	#print accuracy
	print (accuracy_score(pred,labels_test))

if __name__=="__main__": 
        
    workingPart()



