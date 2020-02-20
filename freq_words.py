import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def freq_words(essay_list):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    all_words = []
    essay_lister = []
    data = pd.read_csv('cleanerstill.csv', sep=";")
    
    for es in essay_list:
        essays = [e for e in data[es]]
        
        for i, essay in enumerate(essays):
            tmp = []
            tmp_list = []
            if type(essay) != float:
                splt = essay.split()
                tmp.extend([porter.stem(w) for w in splt if not w in stop_words])
                for w in tmp:
                    if "'" in w:
                        tmp_list.extend(w.split("'"))
                    else:
                        tmp_list.append(w)
                #essay_list.append((data[classifier][i],[word for word in tmp_list if word]))
                all_words.extend(tmp_list)
                
        freq_words = nltk.FreqDist(w for w in all_words)
        word_features = list(freq_words)[:3000]
        
        with open(f"{es}_freq_words.txt", 'w') as file:
            for word in word_features:
                file.write(word+'\n')
        
def main():
    freq_words(['essay0','essay4','essay6','essay7','essay8'])

if '__main__' == __name__:
    main()

