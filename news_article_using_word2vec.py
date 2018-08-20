
import numpy as np
import pandas as pd 
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split




data= pd.read_csv("bbc-text.csv",encoding= 'latin-1')
data.head()
data['category'].value_counts()



def pre_process(text):
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return text





#googles word2vec
from gensim.models import KeyedVectors
filename = 'C:/Users/hp/Desktop/codenote/email_spam_word2vec/GoogleNews-vectors-negative300.bin'
G_w2v = KeyedVectors.load_word2vec_format(filename, binary=True)





print (type(G_w2v))
learned_vocab_G= (G_w2v.vocab)

print (type(learned_vocab_G))


def make_featuers(text):
    
   
    feature= np.zeros((len(text),300),dtype= object)
    c=0;
    for wrd in text:
        if wrd in G_w2v:
            feature[c]=G_w2v[wrd].reshape(1,300)
            c=c+1;
            
    #print(len(feature))
    res= np.zeros(300)
    for i in range(300):
        #print (i)
        res[i]=0
        for j in range(len(feature)):
            res[i]= res[i] + feature[j,i]
    for i in range(300):
        res[i]= res[i]/(len(feature))
    # print (type(feature)) list
   # print(type(res))
    return res
    
txt_fea= data['text'].copy()

#print (type(txt_fea))
#print (type(trial_txt))
txt_fea= txt_fea.apply(pre_process)

print (type(txt_fea))
print(len(txt_fea[0]))
print ("ok")

trial_txt= txt_fea[:]



feat= np.zeros((len(trial_txt),300),dtype=object)
for i in range(len(trial_txt)):
    feat[i]=(make_featuers(trial_txt[i]))
print (len(feat))
print(len(feat[0]))
print ("ok word vectors made") 





train_f,test_f, train_l,test_l= train_test_split(feat,data['category'], test_size=0.25,random_state=111)
print(type(train_i))




from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

svc = SVC(kernel='sigmoid', gamma=1.2)
svc.fit(train_f, train_l)
prediction = svc.predict(test_f)
accuracy_score(test_l,prediction)

