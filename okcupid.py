from random import shuffle
import nltk
import pandas as pd
import pickle
from nltk.stem.porter import PorterStemmer

word_features = []
with open("essay4_freq_words.txt") as file:
    for word in file:
        word_features.append(word)

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word.strip())] = (word.strip() in document_words)
    return features

def main():
    data = pd.read_csv('cleanerstill.csv', sep=";")
    porter = PorterStemmer()
    essay4 = [f for f in data["essay4"][:30000]]
    essay_list = []

    for i, essay in enumerate(essay4):
        tmp = []
        tmp_list = []
        if type(essay) != float:
            tmp.extend([w for w in essay.split()])
            for w in tmp:
                splt = w.split("'")
                for s in splt:
                    if not s.isdigit():
                        tmp_list.append(porter.stem(s))
            essay_list.append((tmp_list, data["sex"][i]))
    
    featuresets = [(document_features(t), f) for (t, f) in essay_list if (t and f)]
    length = len(featuresets)
    shuffle(featuresets)
    train_set, test_set = featuresets[length//2:], featuresets[:length//2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(10)

if __name__ == '__main__':
    main()