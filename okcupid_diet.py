import nltk
import pandas as pd
import pickle
import sys
from collections import defaultdict, Counter
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.scores import precision, recall, f_measure
from nltk.stem.porter import PorterStemmer
from random import shuffle

ngram = int(sys.argv[1])
essay = sys.argv[2]
classifier = sys.argv[3]
all = sys.argv[4]

if ngram == 0:
    if all:
        infile = open(f"Data/all_freq_words", 'rb')
    else:
        infile = open(f"Data/{essay}_freq_words", 'rb')
elif ngram == 1:
    if all:
        infile = open(f"Data/all_freq_bigrams", 'rb')
    else:
        infile = open(f"Data/{essay}_freq_bigrams", 'rb')
else:
    if all:
        infile = open(f"Data/all_freq_trigrams", 'rb')
    else:
        infile = open(f"Data/{essay}_freq_trigrams", 'rb')
freq_ngrams = pickle.load(infile)
word_features = [w for (w, f) in freq_ngrams.most_common(2000)]
infile.close()

infile = open("Data/essay_ngrams", 'rb')
ngram_tuple = pickle.load(infile)
'''tuple with essay_unigrams, essay_bigrams and essay_trigrams
to access unigrams, just ngram_tuple[0]["essay0"]'''
infile.close()

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def bayes(ngram, essay, classifier):
    '''0 = unigram
       1 = bigram
       2 = trigram
    '''
    temp_featuresets = [(document_features(t), "anything" if "anything" in class_dic[classifier] else "other") for (t, class_dic) in ngram_tuple[ngram][essay] if (t and class_dic[classifier] != False)]
    counts = Counter(clas for (text, clas) in temp_featuresets)
    print(counts)
    shuffle(temp_featuresets)
    featuresets = []
    smallest = 100.000
    for key in counts:
        if counts[key] < smallest:
            smallest = key
    counter = counts[smallest]
    for tup in temp_featuresets:
        if tup[1] == smallest:
            featuresets.append(tup)
        elif counter:
            featuresets.append(tup)
            counter -= 1
        else:
            pass

    length = len(featuresets)
    shuffle(featuresets)
    train_set, test_set = featuresets[length//2:], featuresets[:length//2]
    classifier = NaiveBayesClassifier.train(train_set)
    predictions, gold_labels = defaultdict(set), defaultdict(set)

    print('Accuracy:',nltk.classify.accuracy(classifier, test_set))
    for i, (features, label) in enumerate(test_set):
        predictions[classifier.classify(features)].add(i)
        gold_labels[label].add(i)
    for label in predictions:
        print(label, 'Precision:', precision(gold_labels[label], predictions[label]))
        print(label, 'Recall:', recall(gold_labels[label], predictions[label]))
        print(label, 'F1-Score:', f_measure(gold_labels[label], predictions[label]))
        print()
    
    #cm = nltk.ConfusionMatrix(train_set, test_set)
    #print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
    #print('Accuracy:',nltk.classify.accuracy(classifier, test_set))
    #trains = set(classifier)
    #tests = set(test_set)
    #print('Precision:',precision(trains, tests))
    #print('Recall:',recall(trains, tests))
    #print('F-Score:',f_measure(trains, tests))
    classifier.show_most_informative_features(100)

def main():
    bayes(ngram, essay, classifier)

if __name__ == '__main__':
    main()