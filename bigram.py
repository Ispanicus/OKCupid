import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

def freq_bigrams(essay_list)
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    data = pd.read_csv('Data/cleanerstill.csv', sep=";")
    essay_unigrams = {}
    essay_bigrams = {}
    essay_trigrams = {}
    '''essay_unigrams['essay0'] will contain a list of all unigrams for each essay, along with a dictionary of all values for the classifiers
    You access it by doing essay_unigrams['essay0'][i] where i is an index for a tuple of each essay in essay0 and a dictionary of classifier values'''
    classifiers = ["age", "body type", "diet", "drinks", "drugs", "education", "ethnicity", "height", "income", "job", "location", "offspring", "orientation", "pets", "religion", "sex", "sign", "smokes", "speaks", "status"]
    for es in essay_list:
        all_bigrams = []
        essays = [e for e in data[es][:100]]
        essay_bigram_list = []
        essay_trigram_list = []
        for i, essay in enumerate(essays):
            tmp = []
            tmp_list = []
            bigrams_list = []
            classifier_dictionary = {}
            for clas in classifiers:
                if data[clas][i]:
                    classifier_dictionary[clas] = data[clas][i]
                else:
                    
            if type(essay) != float:
                tmp.extend([w for w in essay.split()])
                for w in tmp:
                    splt = w.split("'")
                    for s in splt:
                        if not s.isdigit():
                            tmp_list.append(porter.stem(s))
                for i in range(len(tmp_list)-1):
                    essay_bigram_list.append(" ".join((tmp_list[i],tmp_list[i+1])))
                for i in range(len(tmp_list)-2)
                    essay_trigram_list.append(" ".join((tmp_list[i],tmp_list[i+1],tmp_list[i+2])))
                unigrams_list.append(tmp_list, classifier_dictionary)
                bigrams_list.append((essay_bigram_list, classifier_dictionary))
                trigrams_list.append((essay_trigram_list, classifier_dictionary))
        essay_unigrams[es] = unigrams_list
        essay_bigrams[es] = bigrams_list
        essay_trigrams[es] = trigrams_list

def main():
    freq_bigrams(['essay0','essay4','essay6','essay7','essay8'])

if '__main__' == __name__:
    main()

