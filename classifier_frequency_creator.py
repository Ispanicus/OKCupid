import nltk
import pandas as pd
from math import isnan
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import sys

def freq_creator(data):
    essay_list = ['essay0','essay4']
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    infile = open(data, 'rb')
    data_file = pickle.load(infile)
    infile.close()
    essay_unigrams = {}
    essay_bigrams = {}
    essay_trigrams = {}
    '''essay_unigrams['essay0'] will contain a list of all unigrams for each essay, along with a dictionary of all values for the classifiers
    You access it by doing essay_unigrams['essay0'][i] where i is an index for a tuple of each essay in essay0 and a dictionary of classifier values'''
    classifiers = ["age", "ethnicity", "sex"]
    for es in essay_list:
        all_bigrams = []
        essays = [(idx, e) for idx, e in data_file[es].iteritems()]
        unigrams_list = []
        bigrams_list = []
        trigrams_list = []
        for (i, essay) in essays:
            tmp = []
            tmp_list = []
            essay_bigram_list = []
            essay_trigram_list = [] 
            classifier_dictionary = {}
            for clas in classifiers:
                classifier_dictionary[clas] = data_file[clas][i]
            if type(essay) != float:
                tmp.extend([w for w in essay.split()])
                for w in tmp:
                    splt = w.split("'")
                    for s in splt:
                        if not s.isdigit():
                            tmp_list.append(porter.stem(s))
                for j in range(len(tmp_list)-1):
                    essay_bigram_list.append(" ".join((tmp_list[j],tmp_list[j+1])))
                for k in range(len(tmp_list)-2):
                    essay_trigram_list.append(" ".join((tmp_list[k],tmp_list[k+1],tmp_list[k+2])))
                unigrams_list.append((tmp_list, classifier_dictionary))
                bigrams_list.append((essay_bigram_list, classifier_dictionary))
                trigrams_list.append((essay_trigram_list, classifier_dictionary))
        essay_unigrams[es] = unigrams_list
        essay_bigrams[es] = bigrams_list
        essay_trigrams[es] = trigrams_list
    return (essay_unigrams, essay_bigrams, essay_trigrams)

def main():
    ngrams = freq_creator(sys.argv[1])
    outfile = open(sys.argv[2], 'wb')
    pickle.dump(ngrams, outfile)
    outfile.close()

if '__main__' == __name__:
    main()

