from random import shuffle

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

def main():
    featuresets = [(document_features(t), f) for (f, t) in essay_list if t]
    shuffle(featuresets)
    train_set, test_set = featuresets[1000:], featuresets[:1000]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print(nltk.classify.accuracy(classifier, test_set))
    classifier.show_most_informative_features(100)