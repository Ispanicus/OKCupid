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
    every_word = []
    every_bigram = []
    every_trigram = []
    
    for es in essay_list:
        all_words = []
        all_bigrams = []
        all_trigrams = []
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
                    all_bigrams.append(" ".join((tmp_list[j],tmp_list[j+1])))
                for k in range(len(tmp_list)-2):
                    all_trigrams.append(" ".join((tmp_list[k],tmp_list[k+1],tmp_list[k+2])))
                all_words.extend(tmp_list)
                
        freq_words = nltk.FreqDist(w for w in all_words)
        freq_bigrams = nltk.FreqDist(w for w in all_bigrams)
        freq_trigrams = nltk.FreqDist(w for w in all_trigrams)
        
        with open(f"Data/{es}_freq_words", 'wb') as file:
            pickle.dump(freq_words, file)
        with open(f"Data/{es}_freq_bigrams", 'wb') as file:
            pickle.dump(freq_bigrams, file)
        with open(f"Data/{es}_freq_trigrams", 'wb') as file:
            pickle.dump(freq_trigrams, file)

        every_word.extend(all_words)
        every_bigram.extend(all_bigrams)
        every_trigram.extend(all_trigrams)
        
    freq_all_words = nltk.FreqDist(w for w in every_word)
    freq_all_bigrams = nltk.FreqDist(w for w in every_bigram)
    freq_all_trigrams = nltk.FreqDist(w for w in every_trigram)

    with open("Data/all_freq_words", 'wb') as file:
        pickle.dump(freq_all_words, file)
    with open("Data/all_freq_bigrams", 'wb') as file:
        pickle.dump(freq_all_bigrams, file)
    with open("Data/all_freq_trigrams", 'wb') as file:
        pickle.dump(freq_all_trigrams, file)
        
def main():
    freq_ngrams(['essay0','essay4'])

if '__main__' == __name__:
    main()

