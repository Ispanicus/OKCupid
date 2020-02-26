import nltk
import pandas as pd
import pickle
import sys
from collections import defaultdict
from nltk.classify import NaiveBayesClassifier
from nltk.metrics.scores import precision, recall, f_measure
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import LinearSVC
from random import shuffle

ngram = int(sys.argv[1])
essay = sys.argv[2]
label = sys.argv[3]
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

infile = open("Data/train_ngrams", 'rb')
train_ngrams = pickle.load(infile)
'''tuple with essay_unigrams, essay_bigrams and essay_trigrams
to access unigrams, just ngram_tuple[0]["essay0"]'''
infile.close()

infile = open("Data/dev_ngrams", 'rb')
dev_ngrams = pickle.load(infile)
infile.close()

def label_func(l, cdl):
    if l == "sex" or cdl == "white":
        return cdl
    elif l == "age":
        if cdl <= 30:
            return "u_30"
        else:
            return "o_30"
    else:
        return "n-white"

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def bayes(ngram, essay, label):
    train_features = [(document_features(t), label_func(label, class_dic[label])) for (t, class_dic) in train_ngrams[ngram][essay]]
    dev_features = [(document_features(t), label_func(label, class_dic[label])) for (t, class_dic) in dev_ngrams[ngram][essay]]
    shuffle(train_features)
    shuffle(dev_features)
    training_set, testing_set = train_features, dev_features
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    print("Naive Bayes accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    print(classifier.show_most_informative_features(100))
    predictions, gold_labels = defaultdict(set), defaultdict(set)
    for i, (features, label) in enumerate(testing_set):
        predictions[classifier.classify(features)].add(i)
        gold_labels[label].add(i)
    for label in predictions:
        print(label, 'Precision:', precision(gold_labels[label], predictions[label]))
        print(label, 'Recall:', recall(gold_labels[label], predictions[label]))
        print(label, 'F1-Score:', f_measure(gold_labels[label], predictions[label]))
        print()
    '''
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

    LogisticRegression_classifier = SklearnClassifier(LogisticRegression(max_iter = 150))
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

    LinearSVC_classifier = SklearnClassifier(LinearSVC(max_iter=1200))
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)'''
    return

def main():
    bayes(ngram, essay, label)

if __name__ == '__main__':
    main()