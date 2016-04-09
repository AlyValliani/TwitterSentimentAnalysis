#!/usr/bin/env python3
'''
preprocess.py - this file contains all the necessary functions to pre-process
training/testing files for classification. 

Aly Valliani and Richard Liang
CS65 Final Project
12/18/14
'''

from __future__ import print_function, unicode_literals
import sys

sys.path = [x for x in sys.path if '2.7' not in x]
sys.path.append('/data/cs65/semeval-2015/arktweet/')
sys.path.append('/data/cs65/semeval-2015/scripts/')
sys.path.append('/data/cs65/semeval-2015/jazzy')

from jazzy import spell

import random

import nltk

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from arktweet import tokenize

from parseTweet import parse_tweets
from copy import deepcopy

from nltk import compat
from warnings import warn

'''
tokenizeDataSet - this function uses the CMU ArkTweet to tokenize the tweet
data

Returns: none
'''
def tokenizeDataSet(dataSet):
    tweetsToTokenize = []
    tweets = dataSet['tweets'].keys()
    for tweet in tweets:
        concatTweet = ' '.join(dataSet['tweets'][tweet]['words'])
        tweetsToTokenize.append(concatTweet.lower())
        
    tokenizedTweets = tokenize(tweetsToTokenize)
    for tweet, tokenized in zip(tweets, tokenizedTweets):
        dataSet['tweets'][tweet]['words']  = \
            tokenized.split()

'''
negate - this function adds 'NOT__________' to the beginning of each feature
following the occurrence of a 'not' until the occurrence of a punctuation.

Returns: list containing negated features
'''
def negate(features):
    punctuation = ['.', ',', ';', '?', '!', ':', '"', '\'']
    indices = [i for i, word in enumerate(features) \
            if word.lower() == 'not']
    indices.sort()

    for index in indices:
        counter = index + 1
        numNegated = 0
        while counter < len(features) and \
            all(p not in features[counter] for p in punctuation) and \
            numNegated < 4:
            if 'NOT__________' not in features[counter]:
                features[counter] = 'NOT__________' + features[counter]
                #numNegated += 1
            counter += 1
        if counter < len(features) and numNegated < 4: 
            features[counter] = 'NOT__________' + features[counter]
    return features

'''
shortenURL - this function detects URLs present within tweets and changes them
to 'URL'

Returns: list containing URL shortened features.
'''
def shortenURL(features):
    URLCharacteristics = ['www.', 'http', '//']
    shortenedFeatures = ['URL' if any(characteristic in feature for \
            characteristic in URLCharacteristics) else feature for \
            feature in features]
    return shortenedFeatures

'''
removeHashTagsAndUsernames - this function removes all hash tags and replaces
them with 'AT_USER'

Returns: list of hashtag-less features
'''
def removeHashTagsAndUsernames(features):
    noUserFeatures = ['AT_USER' if feature[0] == '@' else feature for \
            feature in features]
    hashTaglessFeatures = [feature[1:] if feature[0] == '#' else feature \
            for feature in noUserFeatures]
    return  hashTaglessFeatures

'''
replaceAbbreviations - this function uses the abbreviationDict to replace 
abbreviations with their unabbreviated form.

Returns: list of features with abbreviations replaced.
'''
def replaceAbbreviations(features):
    for abbreviation, words in abbreviationDict.items():
        while True:
            try:
                abbrevIndex = features.index(abbreviation)
                features[abbrevIndex:abbrevIndex+1] = words.split()
            except ValueError:
                break
    return features

'''
replaceEmoticons - this function uses the emoticonDict to replace emoticons.

Returns: list of features with emoticons replaced.
'''
def replaceEmoticons(features):
    for emoticon, replacement in emoticonDict.items():
        while True:
            try:
                emoticonIndex = features.index(emoticon)
                features[emoticonIndex:emoticonIndex+1] = replacement
            except ValueError:
                break
    return features

'''
removeStopWords - this function uses the stopWords dictionary to remove stop
words.

Returns: list of features with stop words removed.
'''
def removeStopWords(features):
    return [feature for feature in features if feature not in stopWords]

'''
porterStem - this function uses an nltk stemmer (porter stemmer) to stem all 
words within the feature list.

Returns: list of features with features reduced to their stems.
'''
def porterStem(features):
    porterStemmer = PorterStemmer()
    return [porterStemmer.stem(feature) for feature in features]

'''
lancasterStem - this function uses an nltk stemmer (lancaster stemmer) to stem
all words within the feature list.

Returns: list of features with features reduced to their stems.
'''
def lancasterStem(features):
    lancasterStemmer = LancasterStemmer()
    return [lancasterStemmer.stem(feature) for feature in features]

'''
lemmatize - this function lemmatizes all words within the feature set.

Returns: list of lemmatized features.
'''
def lemmatize(features):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(feature) for feature in features]

'''
fixSpelling - this function uses Jazzy to fix spelling mistakes within the 
feature set

Returns: a list of features with Jazzy-corrected spellings.
'''
def fixSpelling(dataSet):
    tweets = list(dataSet['tweets'].keys())
    totalNumTweets = len(tweets)
    for i in range(0, len(tweets), 20):
        print(i, 'out of', totalNumTweets, 'corrected')
        tweetsToCorrect = tweets[i:i+20]
        words = []
        for tweet in tweetsToCorrect:
            words.append(' '.join(dataSet['tweets'][tweet]['words']))
        correctedWords =  spell(words)
        for tweet, correctText in zip(tweetsToCorrect, correctedWords):
            dataSet['tweets'][tweet]['words'] = correctText.split()

'''
removePunctuation - this function removes common punctuations.

Returns: list of features minus the punctuations.
'''
def removePunctuation(features):
    punctuation = ['.', ',', ';', '?', '!', ':', '"', '\'']
    return [feature for feature in features if all(character not in \
            punctuation for character in feature)]

'''
addPositiveAndNegativeFeatures - this function adds _POS_ and _NEG_ labels
for words that are classified as positive or negative according to the Bing
Liu Lexicon.

Note: The commented code within the function uses MPQA. We didn't use it
since it reduces run-time with no noticeable change in accuracy.

Returns: list containing the features with _POS_ or _NEG_ labels added.
'''
def addPositiveAndNegativeFeatures(features):

    #featuresPOS = features
    #featuresPOS = nltk.pos_tag(featuresPOS)
    positivePlaceholders = ['_POS_' for feature in features if feature in \
            positiveWords]
    negativePlaceholders = ['_NEG_' for feature in features if feature in \
            negativeWords]
    #positivePlaceholdersPOS = []
    #negativePlaceholdersPOS = []
    #weakPositivePlaceholdersPOS = []
    #weakNegativePlaceholdersPOS = []

    '''
    for feature in features:
        if feature in positiveWordsPOS:
            if positiveWordsPOS[feature] == 'anypos':
                positivePlaceholdersPOS.append('_POS_')
            elif (feature, positiveWordsPOS[feature]) in featuresPOS:
                positivePlaceholdersPOS.append('_POS_')
        elif feature in negativeWordsPOS:
            if negativeWordsPOS[feature] == 'anypos':
                negativePlaceholdersPOS.append('_NEG_')
            elif (feature, negativeWordsPOS[feature]) in featuresPOS:
                negativePlaceholdersPOS.append('_NEG_')
        elif feature in weakPositiveWordsPOS:
            if weakPositiveWordsPOS[feature] == 'anypos':
                weakPositivePlaceholdersPOS.append('_POS_')
            elif (feature, weakPositiveWordsPOS[feature]) in featuresPOS:
                weakPositivePlaceholdersPOS.append('_POS_')
        elif feature in weakNegativeWordsPOS:
            if weakNegativeWordsPOS[feature] == 'anypos':
                weakNegativePlaceholdersPOS.append('_NEG_')
            elif (feature, weakNegativeWordsPOS[feature]) in featuresPOS:
                weakNegativePlaceholdersPOS.append('_NEG_')   
    '''
    features.extend(positivePlaceholders)
    features.extend(negativePlaceholders)
    #features.extend(positivePlaceholdersPOS)
    #features.extend(negativePlaceholdersPOS)
    #features.extend(weakPositivePlaceholdersPOS)
    #features.extend(weakNegativePlaceholdersPOS)

    return features

'''
find_bigrams - this function determines bigrams within the feature set.

Returns: list of features with bigrams included.
'''
def find_bigrams(input_list):
      return zip(input_list, input_list[1:])

'''
preprocess - function that utilizes all the pre-processing and feature 
extraction methods on the user-inputted data set(s).

Note: Commented code indicates methods that were tests but not ultimately 
used due to their inability to enhance the performance of our classifier.
'''
def preprocess(trainData): 
    tokenizeDataSet(trainData)
    for tweet in trainData['tweets'].keys():
        features = trainData['tweets'][tweet]['words']
        features = shortenURL(features)
        features = replaceAbbreviations(features)
        features = replaceEmoticons(features)
        features = removeHashTagsAndUsernames(features)
    
        features = addPositiveAndNegativeFeatures(features)
        #features = removeStopWords(features)
        features = porterStem(features) 
        features = negate(features) 
        #features = removePunctuation(features)
        #features = lemmatize(features)
        #features = lancasterStem(features)
        trainData['tweets'][tweet]['words'] = features

    
    #fixSpelling(trainData)

def adjustSentiment(sentimentList):
    adjustedSentiment = None
    if len(sentimentList) == 2 or 'objective' in sentimentList:
        adjustedSentiment = 'neutral'
    else:
        adjustedSentiment = sentimentList[0]

    return adjustedSentiment    
    
def write(trainData, outputFileName):
    outputfile = open(outputFileName, 'w')
    for tweet in trainData['tweets'].keys():
        features = trainData['tweets'][tweet]['words']

        instances = tweet.split('_')
        sentimentList = trainData['tweets'][tweet]['answers']
        toWrite = [instances[0], instances[1], adjustSentiment(sentimentList),\
                ' '.join(features)]
        outputfile.write('\t'.join(toWrite) + '\n')
    
    outputfile.close()

def main():
    usage = 'python3 preprocess.py <INPUT_FILE_NAME> <OUTPUT_FILE_NAME>'
    if len(sys.argv) != 3:
        print(usage)
        return
    
    global abbreviationDict
    abbreviationDict = {}
    for line in  open('abbreviationDict.txt', 'r'):
        fields = line.split(': ')
        abbreviation = fields[0].lower()
        words = fields[1].lower()
        abbreviationDict[abbreviation] = words

   
    global emoticonDict 
    emoticonDict = {}
    for line in  open('emoticonDict.txt', 'r'):
        fields = line.split()
        emoticon = fields[0]
        replacement = fields[1]
        abbreviationDict[emoticon] = replacement

    global stopWords 
    stopWords = []
    for line in open('stopWordsDict.txt', 'r'):
        stopWords.append(line.split()[0])
    
    
    global positiveWords 
    positiveWords = {}
    for line in open('positive-words.txt', 'r', encoding='ISO-8859-1'):
        positiveWords[line.split()[0]] = 1

    global negativeWords 
    negativeWords = {}
    for line in open('negative-words.txt', 'r', encoding='ISO-8859-1'):
        negativeWords[line.split()[0]] = 1

    '''
    #global positiveWords
    #global negativeWords
    #positiveWords = {}
    #negativeWords = {}
    for line in open('unigrams-pmilexicon.txt', 'r'):
        currLine = line.split()
        score = float(currLine[1])

        if score >= 4.5:
            positiveWords[currLine[0]] = score
        elif score <= 4.5:
            negativeWords[currLine[0]] = score
    
    global positiveWordsPOS
    global negativeWordsPOS
    global weakPositiveWordsPOS
    global weakNegativeWordsPOS
    positiveWordsPOS = {}
    negativeWordsPOS = {}
    weakPositiveWordsPOS = {}
    weakNegativeWordsPOS = {}
    for line in open('mpqa.tff', 'r'):
        currLine = line.split()
        wordType = currLine[0]
        POS = currLine[3].split('=')[1]

        if POS == 'noun':
            POS = 'NN'
        elif POS == 'adj':
            POS = 'JJ'
        elif POS == 'verb':
            POS = 'VBD'
        elif POS == 'adverb':
            POS = 'RB'

        if wordType == 'type=strongsubj':
            if currLine[5] == 'priorpolarity=positive':
                positiveWordsPOS[currLine[2]] = POS
            else:
                negativeWordsPOS[currLine[2]] = POS
        else:
            if currLine[5] == 'priorpolarity=negative':
                weakPositiveWordsPOS[currLine[2]] = POS
            else:
                weakNegativeWordsPOS[currLine[2]] = POS
    '''
    trainFileName = sys.argv[1]
    trainData = parse_tweets(trainFileName, 'B')
    preprocess(trainData)
    outputFileName = sys.argv[2]
    write(trainData, outputFileName)

if __name__ == '__main__':
    main()
    
