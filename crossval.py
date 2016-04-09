'''
Aly Valliani, Richard Liang
CS65 Lab 7
11/5/14
'''
import sys
sys.path.append('/data/cs65/semeval-2015/arktweet/')

from arktweet import tokenize
from parseTweet import parse_tweets
from operator import itemgetter
from warmup import createSentimentHistogram
from warmups import *
from math import log
from copy import deepcopy
from arktweet import tokenize 

'''
TweetClassifier - a class to make classifying tweet sentiments all so easy!
'''
class TweetClassifier:

    '''
    Default constructor

    tweetData: dataset to classify
    classifier: what we use to assign/classify sentiments
    stopWords: flag to decide whether we use stop words or not
    caseFolding: flag to decide whether we use case folding or not
    negation: flag to decide whether we use negation or not
    name: name of our classifier
    '''
    def __init__(self, tweetData, classifier, stopWords = False, \
          caseFolding = False, negation = False, name = ''):
        self.stopWords = stopWords
        self.caseFolding = caseFolding
        self.classifier = classifier
        self.negation = negation
        self.tweetData = deepcopy(tweetData)
        self.name = name


    '''
    tokenize - tokenize our tweets!

    Returns nothing
    '''
    def tokenize(self):
        tweetsToTokenize = []
        tweets = self.tweetData['tweets'].keys()
        for tweet in tweets:
            concatTweet = ' '.join(self.tweetData['tweets'][tweet]['words'])
            tweetsToTokenize.append(concatTweet)
        
        tokenizedTweets = tokenize(tweetsToTokenize)
        for tweet, tokenized in zip(tweets, tokenizedTweets):
            self.tweetData['tweets'][tweet]['words']  = \
                tokenized.encode('utf-8').split()

    '''
    crossValidate - perform cross validation and use classifier to 
        assign sentiments.

    Returns a list of n tuples, where n is the number of chunks, 
        of the format (numCorrect, totalClassified)
    '''
    def crossValidate(self):
        tweets = list(self.tweetData['tweets'].keys())
        chunkSize = int(len(tweets) / fold)
        chunks = list(splitIntoChunks(tweets, chunkSize))
    
        if len(chunks) != fold:
            lastChunk = chunks[fold]
            chunks[fold - 1].extend(lastChunk)
            del chunks[-1]
    
        results = []
            
        for index in range(fold):
            trainingSet = []
            testSet = chunks[index]
            for otherIndex in range(fold):
                if otherIndex == index: 
                    continue
                trainingSet.extend(chunks[otherIndex])
            results.append(self.classifier(self.tweetData, trainingSet, \
                testSet, stopWords = self.stopWords, \
                caseFolding = self.caseFolding, \
                negation = self.negation))

        return results

    '''
    analyzeResults - analyze the results of our cross-validation

    results: a list of n tuples, where n is the 
        number of chunks, of the format (numCorrect, totalClassified)
    
    Returns nothing
    '''
    def analyzeResults(self, results):
        totalNumCorrect = 0
        totalResults = 0
        for numCorrect, total in results:
            totalNumCorrect += numCorrect
            totalResults += total
        print '**********', self.name, '**********'
        print 'Num total:', totalResults
        print 'Num correct:', totalNumCorrect
        print 'Accuracy:', float(totalNumCorrect)/totalResults
        print '--------------------------------------'

    '''
    evaluate - perform cross validation and analyze results.

    Returns nothing
    '''
    def evaluate(self):
        results = self.crossValidate()
        self.analyzeResults(results)

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
adjustSentiment - conflat a sentiment if necessary

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
MFS - retrieve most frequent sentment, along with its count from a dataset 
    of tweets.

tweetData: our full tweet dataset
trainingSet: the subset of our full dataset from which we will retrieve the most
    frequent sentiment

Returns a tuple of format (mostFrequentSentiment, count)
'''
def MFS(tweetData, trainingSet):
    sentimentHistogram = createSentimentHistogram(tweetData, trainingSet, True)
    freqList = [(sentiment, count) for sentiment, count in \
        sentimentHistogram.items()]
    freqList.sort(key=itemgetter(1), reverse=True)
    mostFrequentSentiment, freqCount = freqList[0]
    return (mostFrequentSentiment, freqCount)


'''
MFSClassifier - classify using MFS

tweetData: out full dataset
trainingSet: set to train on
testSet: set to test on
stopWords: flag to decide whether we use stop words or not
caseFolding: flag to decide whether we use case folding or not
negation: flag to decide whether we use negation or not

Returns a tuple of format (numCorrect, totalEvaluations)
'''
def MFSClassifier(tweetData, \
        trainingSet, testSet, stopWords = False, \
        caseFolding = False, negation = False):
    mostFrequentSentiment, freqCount = MFS(tweetData, trainingSet)
    total = 0
    numCorrect = 0

    for tweet in testSet:
        total += 1
        actualSentiment = adjustSentiment(tweetData['tweets'][tweet]['answers'])
        if actualSentiment == mostFrequentSentiment:
            numCorrect += 1
        
    return (numCorrect, total)

'''
extractFeatures - modify a list of features based on flags set

features: features to modify
stopWords: flag to decide whether we use stop words or not
caseFolding: flag to decide whether we use case folding or not
negation: flag to decide whether we use negation or not

Returns list of modified features
'''
def extractFeatures(features, stopWords = False, caseFolding = False, \
        negation = False, wordsToFilter = []):
    if caseFolding:
        features = [feature.lower() for feature in features] 
    if stopWords:
        features = [feature for feature in features \
                if feature not in wordsToFilter]
    if negation:
        punctuation = ['.', ',', ';', '?', '!', ':']
        indices = [i for i, word in enumerate(features) \
                if word.lower() == 'not']
        indices.sort()

        for index in indices:
            counter = index + 1
            while counter < len(features) and \
                all(p not in features[counter] for p in punctuation):
                if 'NOT__________' not in features[counter]:
                    features[counter] = 'NOT__________' + features[counter]
                counter += 1
            if counter < len(features): 
                features[counter] = 'NOT__________' + features[counter]

    return features

'''
createDecisionList - create a decision list

tweetData: out full tweet dataset
trainngSet: the set to train on
stopWords: flag to decide whether we use stop words or not
caseFolding: flag to decide whether we use case folding or not
negation: flag to decide whether we use negation or not

Returns our decision list
'''
def createDecisionList(tweetData, trainingSet, stopWords = False, \
        caseFolding = False, negation = False):
    sentimentFeaturesListMap = {}
    featuresSentimentsMap = {}
    wordsToFilter = findStopWords(tweetData, trainingSet, caseFolding)
    for tweet in trainingSet:
        features = extractFeatures(list(tweetData['tweets'][tweet]['words']), \
                stopWords, caseFolding, negation, wordsToFilter)
        adjustedSentiment = adjustSentiment(tweetData['tweets'][tweet] \
                ['answers'])
        for feature in features:
            if feature not in featuresSentimentsMap:
                featuresSentimentsMap[feature] = [adjustedSentiment]
            else:
                if adjustedSentiment not in featuresSentimentsMap[feature]:
                    featuresSentimentsMap[feature].append(adjustedSentiment)
        
        if adjustedSentiment in sentimentFeaturesListMap:
            sentimentFeaturesListMap[adjustedSentiment].extend(features)
        else:
            sentimentFeaturesListMap[adjustedSentiment] = features

    sentimentsFeaturesHistogramMap = {}
    for sentiment, featuresList in sentimentFeaturesListMap.items():
        featureHistogram = countWords(featuresList)
        sentimentsFeaturesHistogramMap[sentiment] = featureHistogram            

    decisionList = []
    for sentiment, featureHistogram in \
        sentimentsFeaturesHistogramMap.items():
        for feature, count in featureHistogram.items():
            numOccurInOtherSentiments = 0.0
            for otherSent in [otherSentiment for otherSentiment in \
                featuresSentimentsMap[feature] if otherSentiment != sentiment]:
                numOccurInOtherSentiments += \
                    sentimentsFeaturesHistogramMap[otherSent][feature]

            score = log((float(count + 0.1) / \
                        (numOccurInOtherSentiments + 0.1)), 2)
            
            if score > 0:
                decisionList.append((feature, sentiment, score))

    decisionList.sort(key=itemgetter(2), reverse=True)
    return decisionList

'''
decisionListClassifier - classify using decisionLst

tweetData: out full dataset
trainingSet: set to train on
testSet: set to test on
stopWords: flag to decide whether we use stop words or not
caseFolding: flag to decide whether we use case folding or not
negation: flag to decide whether we use negation or not

Returns a tuple of format (numCorrect, totalEvaluations)
'''
def decisionListClassifier(tweetData, trainingSet, \
        testSet, stopWords = False, \
        caseFolding = False, negation = False):
    decisionList  = createDecisionList(tweetData, \
            trainingSet, stopWords, caseFolding, negation)
    mostFrequentSentiment, count = MFS(tweetData, trainingSet)
    numCorrect, total = 0, 0
    for tweet in testSet:
        total += 1
        features = extractFeatures(list(tweetData['tweets'][tweet]['words']), \
                stopWords, caseFolding, negation, wordsToFilter = []) 
        assignedSentiment = None
        for feature, sentiment, score in decisionList:
            if feature in features:
                assignedSentiment = sentiment
                break

        if assignedSentiment == None:
            assignedSentiment = mostFrequentSentiment

        actualSentiment = adjustSentiment(tweetData['tweets'][tweet]['answers'])
        if actualSentiment == assignedSentiment:
            numCorrect += 1
    
    return (numCorrect, total)

'''
naiveBayesClassifier - classify using naiveBayes

tweetData: out full dataset
trainingSet: set to train on
testSet: set to test on
stopWords: flag to decide whether we use stop words or not
caseFolding: flag to decide whether we use case folding or not
negation: flag to decide whether we use negation or not

Returns a tuple of format (numCorrect, totalEvaluations)
'''
def naiveBayesClassifier(tweetData, trainingSet, testSet, \
        stopWords = False, caseFolding = False, negation = False):
    sentimentFeaturesListMap = {}
    sentiments = ['neutral', 'positive', 'negative']
    sentimentTweetHistogram = createSentimentHistogram(tweetData, \
            trainingSet, True)
    sentimentFeaturesInstanceHistogram = {}
    wordsToFilter = findStopWords(tweetData, trainingSet, caseFolding)
    
    for tweet in trainingSet:
        adjustedSentiment = adjustSentiment \
            (list(tweetData['tweets'][tweet]['answers']))
        features = extractFeatures(list(tweetData['tweets'][tweet]['words']), \
                stopWords, caseFolding, negation, wordsToFilter)
 
        if adjustedSentiment in sentimentFeaturesListMap:
            sentimentFeaturesListMap[adjustedSentiment].extend(features)
        else:
            sentimentFeaturesListMap[adjustedSentiment] = features

        if adjustedSentiment in sentimentFeaturesInstanceHistogram:
            sentimentFeaturesInstanceHistogram[adjustedSentiment] += \
                len(features)
        else:
            sentimentFeaturesInstanceHistogram[adjustedSentiment] = \
                len(features)

    sentimentsFeaturesHistogramMap = {}
    for sentiment, featuresList in sentimentFeaturesListMap.items():
        featureHistogram = countWords(featuresList)
        sentimentsFeaturesHistogramMap[sentiment] = featureHistogram
    
    total, numCorrect = 0, 0
    for tweet in testSet:
        total += 1
        results = []
        features = extractFeatures(list(tweetData['tweets'][tweet]['words']), \
                stopWords, caseFolding, negation, wordsToFilter)
        for sentiment in sentiments:
            score = float(sentimentTweetHistogram[sentiment]) / len(trainingSet)
            for feature in features:
                if feature in sentimentsFeaturesHistogramMap[sentiment]:
                    numOccur = sentimentsFeaturesHistogramMap[sentiment] \
                        [feature] + 0.1
                else:
                    numOccur = 0.1
                score *= float(numOccur) / \
                    (sentimentFeaturesInstanceHistogram[sentiment] + 0.1)

            results.append((sentiment, score))
        results.sort(key = itemgetter(1), reverse = True)
        desiredSentiment = results[0][0]

        actualSentiment = adjustSentiment(list(tweetData['tweets'][tweet] \
            ['answers']))
        if actualSentiment == desiredSentiment:
            numCorrect += 1

    return (numCorrect, total)

'''
testData - quick unit test of our data functionality

tweetData: our full tweet dataset

Returns nothing
'''
def testData(tweetData):
    sentimentFeaturesListMap = {}
    sentimentFeaturesHistogramMap = {}
    actualNumFeatures = 0
    for tweet in tweetData['tweets'].keys():
        adjustedSentiment = adjustSentiment \
            (list(tweetData['tweets'][tweet]['answers']))
        features = extractFeatures(list(tweetData['tweets'][tweet]['words']))
        actualNumFeatures += len(features)
        if adjustedSentiment in sentimentFeaturesListMap:
            sentimentFeaturesListMap[adjustedSentiment].extend(features)
        else:
            sentimentFeaturesListMap[adjustedSentiment] = features

    sentimentsFeaturesHistogramMap = {}
    for sentiment, featuresList in sentimentFeaturesListMap.items():
        featureHistogram = countWords(featuresList)
        sentimentsFeaturesHistogramMap[sentiment] = featureHistogram

    featuresInHistogram = 0
    for sentiment, featureHistogram in sentimentsFeaturesHistogramMap.items():
        for feature, count in featureHistogram.items():
            featuresInHistogram += count

    assert(actualNumFeatures == featuresInHistogram)
    print 'Tests Passed'

'''
findStopWords - find stop words using trainingSet

tweetData: our full tweet dataset
trainingSet: the set to find stop words on
caseFolding: optional param to lowercase all words

Returns list of stopWords
'''
def findStopWords(tweetData, trainingSet, caseFolding = False):
    words = []
    for tweet in trainingSet:
        toAdd = list(tweetData['tweets'][tweet]['words'])
        if caseFolding:
            toAdd = [word.lower() for word in toAdd]
        words.extend(toAdd)

    frequencies = wordsByFrequency(words)
    return [word for word, frequency in frequencies[0:50]]

def numberFive(tweetData): 
    tweets = list(tweetData['tweets'].keys())
    chunkSize = int(len(tweets) / fold)
    chunks = list(splitIntoChunks(tweets, chunkSize))
    if len(chunks) != fold:
        lastChunk = chunks[fold]
        chunks[fold - 1].extend(lastChunk)
        del chunks[-1]

    for i in range(len(chunks)):
        mostFrequentSentiment, freqCount = MFS(tweetData, chunks[i])
        print 'Most frequent sentiment for chunk', i, 'is:', \
                mostFrequentSentiment, 'with count:', freqCount
    
def numberSix(tweetData):
    plainMFSClassifier = TweetClassifier(tweetData, \
            MFSClassifier, name = 'MFS Classifier')
    plainMFSClassifier.evaluate()

def numberSeven(tweetData):
    normalDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = False, \
        negation = False, name = 'Normal DL')
    normalDecListClassifier.evaluate()
    
    stopWordsDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = True, caseFolding = False, \
        negation = False, name = 'DL with Stop Words')
    stopWordsDecListClassifier.evaluate()
    
    caseFoldDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = True, \
        negation = False, name = 'DL with Case Folding')
    caseFoldDecListClassifier.evaluate()

def numberEight(tweetData):
    negationDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = False, \
        negation = True, name = 'DL with Negation')
    negationDecListClassifier.evaluate()

def numberNine(tweetData):
    normalBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = False, \
        negation = False, name = 'Normal Naive Bayes')
    normalBayesClassifier.evaluate()

def numberTen(tweetData):
    stopWordsBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = False, \
        negation = False, name = 'Naive Bayes with Stop Words')
    stopWordsBayesClassifier.evaluate()

    caseFoldingBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = True, \
        negation = False, name = 'Naive Bayes with Case Folding')
    caseFoldingBayesClassifier.evaluate()

    negationBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = False, \
        negation = True, name = 'Naive Bayes with Negation')
    negationBayesClassifier.evaluate()

    negationStopBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = False, \
        negation = True, name = 'Naive Bayes with Negation and Stop Words')
    negationStopBayesClassifier.evaluate()

    negationCaseBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = True, \
        negation = True, name = 'Naive Bayes with Negation and Case Folding')
    negationCaseBayesClassifier.evaluate()

    stopCaseBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = True, \
        negation = False, name = 'Naive Bayes with Stop Words and Case Folding')
    stopCaseBayesClassifier.evaluate()

    allBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = True, \
        negation = True, name = 'Naive Bayes with Everything')
    allBayesClassifier.evaluate()

def numberEleven(tweetData):
    normalBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = False, \
        negation = False, name = 'Tokenized Naive Bayes Tokenization')
    normalBayesClassifier.tokenize()
    normalBayesClassifier.evaluate()
    
    stopWordsBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = False, \
        negation = False, name = 'Tokenized Naive Bayes with Stop Words')
    stopWordsBayesClassifier.tokenize()
    stopWordsBayesClassifier.evaluate()

    caseFoldingBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = True, \
        negation = False, name = 'Tokenized Naive Bayes with Case Folding')
    caseFoldingBayesClassifier.tokenize()
    caseFoldingBayesClassifier.evaluate()

    negationBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = False, \
        negation = True, name = 'Tokenized Naive Bayes with Negation')
    negationBayesClassifier.tokenize()
    negationBayesClassifier.evaluate()

    negationStopBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = False, \
        negation = True, name = 'Tokenized Naive Bayes with Negation and Stop Words')
    negationStopBayesClassifier.tokenize()
    negationStopBayesClassifier.evaluate()

    negationCaseBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = False, caseFolding = True, \
        negation = True, name = 'Tokenized Naive Bayes with Negation and Case Folding')
    negationCaseBayesClassifier.tokenize()
    negationCaseBayesClassifier.evaluate()

    stopCaseBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = True, \
        negation = False, name = 'Tokenized Naive Bayes with Stop Words and Case Folding')
    stopCaseBayesClassifier.tokenize()
    stopCaseBayesClassifier.evaluate()

    allBayesClassifier = TweetClassifier(tweetData, \
        naiveBayesClassifier, stopWords = True, caseFolding = True, \
        negation = True, name = 'Tokenized Naive Bayes with Everything')
    allBayesClassifier.tokenize()
    allBayesClassifier.evaluate()
    
    normalDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = False, \
        negation = False, name = 'Tokenized DL')
    normalDecListClassifier.tokenize()
    normalDecListClassifier.evaluate()

    stopWordsDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = True, caseFolding = False, \
        negation = False, name = 'Tokenized DL with Stop Words')
    stopWordsDecListClassifier.tokenize()
    stopWordsDecListClassifier.evaluate()

    CFDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = True, \
        negation = False, name = 'Tokenized DL with Case Folding')
    CFDecListClassifier.tokenize()
    CFDecListClassifier.evaluate()
    
    negDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = False, \
        negation = True, name = 'Tokenized DL with Negation')
    negDecListClassifier.tokenize()
    negDecListClassifier.evaluate()

    SWCFDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = True, caseFolding = True, \
        negation = False, name = 'Tokenized DL with Stop Words and Case Folding')
    SWCFDecListClassifier.tokenize()
    SWCFDecListClassifier.evaluate()

    SWNDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = True, caseFolding = False, \
        negation = True, name = 'Tokenized DL with Stop Words and Negation')
    SWNDecListClassifier.tokenize()
    SWNDecListClassifier.evaluate()

    CFNDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = False, caseFolding = True, \
        negation = True, name = 'Tokenized DL with Case Folding and Negation')
    CFNDecListClassifier.tokenize()
    CFNDecListClassifier.evaluate()

    allDecListClassifier = TweetClassifier(tweetData, \
        decisionListClassifier, stopWords = True, caseFolding = True, \
        negation = True, name = 'Tokenized DL with Everything')
    allDecListClassifier.tokenize()
    allDecListClassifier.evaluate()

def main():
    filename = '/data/cs65/semeval-2015/B/train/twitter-train-full-B.tsv'
   
    global fold
    fold = 5
    
    tweetData = parse_tweets(filename, 'B')
    testData(tweetData)   
    #numberFive(tweetData)
    #numberSix(tweetData)
    #numberSeven(tweetData) 
    #numberEight(tweetData)
    #numberNine(tweetData)
    #numberTen(tweetData)
    #numberEleven(tweetData)

if __name__ == '__main__':
    main()
