from random import shuffle
import nltk
import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer

infile = open("Data/essay4_freq_bigrams", 'rb')
freq_bigrams = pickle.load(infile)
word_features = [w for (w, f) in freq_bigrams.most_common(1500)]
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

def main():
    bigram_4 = ngram_tuple[1]['essay4']
    featuresets = [(document_features(t), class_dic["sex"]) for (t, class_dic) in bigram_4 if (t and class_dic["sex"] != 'nan')]
    length = len(featuresets)
    shuffle(featuresets)
    train_set, test_set = featuresets[length//2:], featuresets[:length//2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(100)

if __name__ == '__main__':
    main()