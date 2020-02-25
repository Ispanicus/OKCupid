import nltk
import pandas as pd
import pickle
import sys
from collections import defaultdict
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.scores import precision, recall, f_measure
from nltk.stem.porter import PorterStemmer
from random import shuffle

essay = sys.argv[1]
classifier = sys.argv[2]
all = sys.argv[3]
if all:
    infile = open(f"Data/all_freq_ngrams", 'rb')
else:
    infile = open(f"Data/{essay}_freq_ngrams", 'rb')
freq_ngrams = pickle.load(infile)
word_features = [w for (w, f) in freq_ngrams.most_common(2000)]
infile.close()
infile = open("Data/essay_ngrams", 'rb')
ngram_tuple = pickle.load(infile)
'''tuple with essay_unigrams, essay_bigrams and essay_trigrams
to access unigrams, just ngram_tuple["essay0"]'''
infile.close()

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def bayes(essay, classifier):
    featuresets = [(document_features(t), class_dic[classifier]) for (t, class_dic) in ngram_tuple[essay][:len(ngram_tuple[essay])//2] if (t and class_dic[classifier] != False and type(class_dic[classifier]) != float)]
    featuresets2 = [(document_features(t), class_dic[classifier]) for (t, class_dic) in ngram_tuple[essay][len(ngram_tuple[essay])//2:] if (t and class_dic[classifier] != False and type(class_dic[classifier]) != float)]
    featuresets.extend(featuresets2)
    length = len(featuresets)
    shuffle(featuresets)
    train_set, test_set = featuresets[length//10:], featuresets[:length//10]
    classifier = NaiveBayesClassifier.train(train_set)
    '''predictions, gold_labels = defaultdict(set), defaultdict(set)
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
    #print('F-Score:',f_measure(trains, tests))'''
    classifier.show_most_informative_features(100)
    return

def main():
    bayes(essay, classifier)

if __name__ == '__main__':
    main()