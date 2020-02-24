import nltk
import pandas as pd
import pickle
import sys
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
    featuresets = [(document_features(t), class_dic[classifier]) for (t, class_dic) in ngram_tuple[ngram][essay] if (t and (class_dic[classifier] != (' ' or False)))]
    length = len(featuresets)
    shuffle(featuresets)    
    train_set, test_set = featuresets[length//2:], featuresets[:length//2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    return classifier.show_most_informative_features(100)
    

def main():
    print(bayes(ngram, essay, classifier))

if __name__ == '__main__':
    main()