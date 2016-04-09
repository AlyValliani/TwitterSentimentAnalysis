#!/usr/bin/env python3
'''
classifier.py - this file contains our logistic regression classifier which
was submitted as part of SemEval-2015 Task 10 Subtask B. Performs 5-fold
cross validation on training data or direct training and testing when given
two different tweet files.

Aly Valliani and Richard Liang
CS65 Final Project
12/18/14
'''

from __future__ import print_function, unicode_literals
import sys

sys.path = [x for x in sys.path if '2.7' not in x]
sys.path.append('/data/cs65/semeval-2015/arktweet/')
sys.path.append('/data/cs65/semeval-2015/scripts/')

import random

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import *
from sklearn.feature_selection import *

from arktweet import tokenize, dict_tagger
from scorer import scorer
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from nltk.classify.scikitlearn import SklearnClassifier
from parseTweet import parse_tweets
from copy import deepcopy

from nltk.classify.api import ClassifierI
from nltk.probability import DictionaryProbDist
from nltk import compat
from warnings import warn

'''
TweetClassifier - a class to make classifying tweet sentiments all so easy!
'''
class TweetClassifier:

    MODE_CROSS_VALIDATE = "mode_cross_validate"
    MODE_TEST_FILE = "mode_test_file"    
    
    '''
    Default constructor

    trainingData: dataset to train on
    testData: data to test on
    classifier: classifier (nltk.classifiy) to use.
    name: name of classifier 
    fold number of folds for cross validation
    '''
    def __init__(self, trainingData, testData, classifier, name='', fold = -1):
        self.trainingData = deepcopy(trainingData)
        self.testData = deepcopy(testData)
        self.classifier = classifier
        self.name = name
        self.fold = fold


    '''
    crossValidate - perform cross validation and use classifier to 
        assign sentiments.

    Returns a list of n tuples, where n is the number of chunks, 
        of the format (numCorrect, totalClassified)
    '''
    def crossValidate(self):
        tweets = list(self.trainingData['tweets'].keys())
        chunkSize = int(len(tweets) / self.fold)
        chunks = list(splitIntoChunks(tweets, chunkSize))
   
        if len(chunks) != self.fold:
            lastChunk = chunks[self.fold]
            chunks[self.fold - 1].extend(lastChunk)
            del chunks[-1]
    
        results = []   
        for index in range(self.fold):
            trainingSet = []
            testSet = chunks[index]
            for otherIndex in range(self.fold):
                if otherIndex == index: 
                    continue
                trainingSet.extend(chunks[otherIndex]) 
            results.append(self.scoreClassify(trainingSet, testSet))

        return results

    '''
    scoreClassify - given a training and test set, train on the training set
        and test on the test set.

    trainingSet: the set to train on
    testSet: the set to test on

    Returns a tuple of the format (numCorrect, totalClassified)
    '''
    def scoreClassify(self, trainingSet, testSet):
        wrapper = SklearnDenseClassifier(self.classifier(dual = True, \
                class_weight = 'auto'))
        labels = ['positive', 'neutral', 'negative']
        trainingFeatureSets = formatTweetData(trainingSet, self.trainingData)
        testFeatureSets = formatTweetData(testSet, self.testData)

        wrapper.train(trainingFeatureSets)
        resultingLabels = wrapper.classify_many([featureSet for featureSet, \
                label in testFeatureSets])
        
        actualLabels = [label for featureSet, label in testFeatureSets]
        
        confusionTable = {}
        for label in labels:
            confusionTable[label] = {}
            for otherLabel in labels:
                confusionTable[label][otherLabel] = 0

        for resultLabel, actualLabel in zip(resultingLabels, actualLabels):
            confusionTable[resultLabel][actualLabel] += 1
    
        return scorer(confusionTable)

    '''
    evaluateTestFile - calls formating functions on the tweet data and then
    outputs tab-separated instance IDs and their class predictions.
    '''
    
    def evaluateTestFile(self):
        wrapper = SklearnDenseClassifier(self.classifier(dual = True, \
                class_weight = 'auto'))
        labels = ['positive', 'neutral', 'negative']
        tweetsToTrain = self.trainingData['tweets'].keys()
        tweetsToTest = self.testData['tweets'].keys()

        trainingFeatureSets = formatTweetData(tweetsToTrain, self.trainingData)
        testFeatureSets = formatTweetData(tweetsToTest, self.testData)

        wrapper.train(trainingFeatureSets)
        resultingLabels = wrapper.classify_many([featureSet for featureSet, \
                label in testFeatureSets])

        for instanceID, label in zip(tweetsToTest, resultingLabels):
            print(instanceID + '\t' + label)

    '''
    evaluate - perform cross validation and analyze results.
    
    mode: the mode of the evaluation, either MODE_CROSS_VALIDATE or 
        MODE_TEST_FILE

    Returns nothing
    '''
    def evaluate(self, mode):
        if mode == TweetClassifier.MODE_CROSS_VALIDATE:
            results = self.crossValidate()
            print('\n*****Average =', sum(results) / len(results), '\n*****')
        elif mode == TweetClassifier.MODE_TEST_FILE:
            self.evaluateTestFile()

'''
SklearnDenseClassifier - class used to make data denser to improve classifier
performance.
'''
       
class SklearnDenseClassifier(SklearnClassifier):
    
    '''
    train - overrides train method to utilize denser data.
    '''
    def train(self, labeled_featuresets):
        X, y = list(compat.izip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X) 
        X = X.toarray()
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)

        return self
 
    '''
    classify_many - overrides classify method to utilize dense data.

    Returns: class predictions
    '''
    def classify_many(self, featuresets):
        X = self._vectorizer.transform(featuresets) 
        X = X.toarray()
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)] 


'''
splitIntoChunks - split a list of elements into chunks of a specified size

lst: the list to split
chunkSize: the size of the chunks in the split

Returns a list of chunks, with each chunk being size 
    chunkSize or len(lst) % chunkSize
'''
def splitIntoChunks(lst, chunkSize):
    for i in range(0, len(lst), chunkSize):
        yield lst[i : i + chunkSize]

'''
adjustSentiment - conflate a sentiment if necessary

sentimentList: the list of sentiments to conflate, if necessary.

Returns conflated sentiment, if we conflated, or the only sentiment in the list.
'''
def adjustSentiment(sentimentList):
    adjustedSentiment = None
    if len(sentimentList) == 2 or 'objective' in sentimentList:
        adjustedSentiment = 'neutral'
    else:
        adjustedSentiment = sentimentList[0]

    return adjustedSentiment


'''
formatTweetData - given a list of tweet ids, format those tweets into
    a feature list we can use for our nltk classifier

tweets: the ids for the tweets to format
tweetData: dataset containing the tweet text

Returns a featurelist with elements of the format (featureset, label),
    where featureset(one per tweet)  is a feature->count map and label is the
    sentiment.
'''
def formatTweetData(tweets, tweetData):
    featureList = []
    for tweet in tweets:
        words = list(tweetData['tweets'][tweet]['words'])
        featureSet = {}
        for word in words:
            feature = word
            if feature in featureSet:
                featureSet[feature] += 1
            else:
                featureSet[feature] = 1
                
        label = adjustSentiment(tweetData['tweets'][tweet]['answers'])
        featureList.append((featureSet, label))
    return featureList
    
def main():
    usage = 'python3 classifier.py training_file test_file OR \n \
        python3 classifier.py training_file num_folds_for_cross_val'
    if len(sys.argv) != 3:
        print(usage)
        return

    trainFileName = sys.argv[1]
    trainData = parse_tweets(trainFileName, 'B')
    
    try:
        fold = int(sys.argv[2])
        
        '''
        linearSVCClassifier = TweetClassifier(trainData, trainData, \
            LinearSVC, name = 'Linear SVC Classifier', fold = fold)
        linearSVCClassifier.evaluate(TweetClassifier.MODE_CROSS_VALIDATE)
        '''

        logisticRegClassifier = TweetClassifier(trainData, trainData, \
            LogisticRegression, name = 'Logistic Regression Classifier', \
            fold = fold)
        logisticRegClassifier.evaluate(TweetClassifier.MODE_CROSS_VALIDATE) 
    
    except ValueError:
        testFileName = sys.argv[2]
        testData = parse_tweets(testFileName, 'B') 
        '''
        linearSVCClassifier = TweetClassifier(trainData, testData, \
            LinearSVC, name = 'Linear SVC Classifier')
        linearSVCClassifier.evaluate(TweetClassifier.MODE_TEST_FILE)
        '''
        logisticRegClassifier = TweetClassifier(trainData, testData, \
            LogisticRegression, name = 'Logistic Regression  Classifier')
        logisticRegClassifier.evaluate(TweetClassifier.MODE_TEST_FILE) 

if __name__ == '__main__':
    main()
