
# coding: utf-8

# ## Build unigram dictionary

# In[1]:

import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams

sentences=['a b a','b a a b','a a a','b a b b','b b a b','a a a b'] # data 

unigrams=[]

for elem in sentences:
   unigrams.extend(elem.split())
  
from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
   unigram_counts[word]/=unigram_total

print(unigram_counts)


# ## Build bigram dictionary

# In[2]:

def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent.split(),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]/=tot_count
     
    return model

bigram_counts= bigram_model(sentences)
print(bigram_counts)


# ## Build trigram dictionary

# In[3]:

def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent.split(),3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]/=tot_count
     
    return model

trigram_counts= trigram_model(sentences)
print(trigram_counts)


# ## Test Scores of each model

# In[4]:

test_sentences=['a b a b','b a b a','a b b','b a a a a a b','a a a','b b b b a']

import numpy as np

test_unigram_arr=[]

print('Unigram test probabilities\n')
for elem in test_sentences:
    p_val=np.prod([unigram_counts[i] for i in elem.split()])
    test_unigram_arr.append(p_val)
    print('The sequence '+elem+' has unigram probablity of '+ str(round(p_val,4)))


print('\nBigram test probabilities\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=1
    for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
        p_val*=bigram_counts[w1][w2]
    print('The sequence '+ elem +' has bigram probablity of '+ str(round(p_val,4)))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test probabilities\n')
for elem in test_sentences:
    p_val=1
    for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
        try:
            p_val*=trigram_counts[(w1,w2)][w3]
        except Exception as e:
            p_val=0
            break
    print('The sequence '+ elem +' has trigram probablity of '+ str(round(p_val,4)))
    
    test_trigram_arr.append(p_val)
            


# In[7]:

import matplotlib.pyplot as plt

x_axis=[i for i in range(1,4)]

y_axis=[np.mean(test_unigram_arr), np.mean(test_bigram_arr), np.mean(test_trigram_arr)]

plt.scatter(x_axis,y_axis)
plt.show()


# In[ ]:



