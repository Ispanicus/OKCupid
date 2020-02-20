import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

def freq_words(essay_list):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    data = pd.read_csv('cleanerstill.csv', sep=";")
    every_word = []
    
    for es in essay_list:
        all_words = []
        essays = [e for e in data[es][:30000]]
        
        for i, essay in enumerate(essays):
            tmp = []
            tmp_list = []
            if type(essay) != float:
                tmp.extend([porter.stem(w) for w in essay.split() if not w in stop_words])
                for w in tmp:
                    splt = w.split("'")
                    for s in splt:
                        if not s.isdigit() and s not in stop_words:
                            tmp_list.append(s)
                #essay_list.append((data[classifier][i],[word for word in tmp_list if word]))
                all_words.extend(tmp_list)
                
        freq_words = nltk.FreqDist(w for w in all_words)
        word_features = list(freq_words)[:1500]
        
        with open(f"{es}_freq_words", 'wb') as file:
            pickle.dump(freq_words, file)
        
        #list of ALL words throughout all essays
        every_word.extend(all_words)
        
    freq_all = nltk.FreqDist(w for w in all_words)
    all_features = list(freq_all)[:1500]
    with open("all_freq_words", 'wb') as file:
        pickle.dump(freq_all, file)
        
def main():
    freq_words(['essay0','essay4','essay6','essay7','essay8'])

if '__main__' == __name__:
    main()

