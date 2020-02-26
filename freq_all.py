import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
import pickle

def freq_ngrams(essay_list):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    infile = open("Data/train_data", 'rb')
    train_data = pickle.load(infile)
    every_ngram = []
    
    for es in essay_list:
        all_ngrams = []
        essays = [(idx, e) for idx, e in train_data[es].iteritems()]
        for i, essay in essays:
            tmp = []
            tmp_list = []
            if type(essay) != float:
                tmp.extend([porter.stem(w) for w in essay.split() if not w in stop_words])
                for w in tmp:
                    splt = w.split("'")
                    for s in splt:
                        if not s.isdigit() and s not in stop_words:
                            tmp_list.append(s)
                for j in range(len(tmp_list)-1):
                    all_ngrams.append(" ".join((tmp_list[j],tmp_list[j+1])))
                for k in range(len(tmp_list)-2):
                    all_ngrams.append(" ".join((tmp_list[k],tmp_list[k+1],tmp_list[k+2])))
                all_ngrams.extend(tmp_list)
                
        freq_ngrams = nltk.FreqDist(w for w in all_ngrams)
        
        with open(f"Test/{es}_freq_ngrams", 'wb') as file:
            pickle.dump(freq_ngrams, file)

        every_ngram.extend(all_ngrams)
        
    freq_all_ngrams = nltk.FreqDist(w for w in every_ngram)

    with open("Test/all_freq_ngrams", 'wb') as file:
        pickle.dump(freq_all_ngrams, file)
        
def main():
    freq_ngrams(['essay0','essay4'])

if '__main__' == __name__:
    main()

