import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def freq_words(essaynr, classifier):
    stop_words = stopwords.words('english')
    porter = PorterStemmer()
    all_words = []
    essay_list = []
    essay_lister = []
    data = pd.read_csv('cleanerstill.csv', sep=";")
    essays = [essay for essay in data[essaynr]]
    
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
            essay_list.append((data[classifier][i],[word for word in tmp_list if word]))
            all_words.extend(tmp_list)
            
    freq_words = nltk.FreqDist(w for w in all_words)
    word_features = list(freq_words)[:3000]
    
    with open(f"{essaynr}_freq_words.txt", 'w') as file:
        for word in word_features:
            file.write(word+'\n')
    
    with open(f"{essaynr}_wordlist_{classifier}.txt", 'w') as file:
        for e in essay_list:
            file.write(e,'\n')

def main():
    freq_words("essay4", "sex")

if '__main__' == __name__:
    main()

