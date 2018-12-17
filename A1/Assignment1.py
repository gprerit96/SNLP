
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams


# In[2]:


sentences=brown.sents()
sentences=sentences[0:40000]


   


# In[13]:


from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
 
print(unigram_counts)


# In[3]:


str_sent=[]
for elem in sentences:
    str_elem=""
    str_elem=str_elem+elem[0].lower()
    for word in elem[1:]:
        word=word.lower()
        str_elem=str_elem+" "+word
    str_sent.append(str_elem)
print(str_sent)


# In[15]:



tokenizer = RegexpTokenizer(r'\w+')

unigrams=[]
sents = []
for elem in sentences:
    i=0
    st = ""
    while(i<(len(elem)-1)):
        st = st + elem[i].lower() + " "
        i=i+1
    st = st + elem[i].lower()
    #print (st)
    sents.append(st)
    
for elem in sents:
    elem_w = tokenizer.tokenize(elem)
    unigrams.extend(elem_w)

from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
    
print(unigram_counts)


# In[16]:


print(str_sent[-1])


# In[17]:


i=0
while(True):
    if (sents[i]!=str_sent[i]):
        print(sents[i])
        print(str_sent[i])
    i=i+1
    if(i>=len(str_sent)):
        break


# In[5]:


from nltk.tokenize import RegexpTokenizer
tokenizer=RegexpTokenizer(r'\w+')
unigrams=[]

for elem in str_sent:
    elem_tokens=tokenizer.tokenize(elem)
    unigrams.extend(elem_tokens)
from collections import Counter
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
 
print(unigram_counts)
N=len(unigram_counts)


# In[7]:


import operator
sorted_unigram = list(reversed(sorted(unigram_counts.items(), key=operator.itemgetter(1))))


# In[15]:


import math
math.log(5)


# In[18]:


N=len(sorted_unigram)
i=1
r=[]
while(i<=N):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1
    
f=[math.log(y) for (x,y) in sorted_unigram]


# In[20]:


import matplotlib.pyplot as plt
plt.plot(f,r)


# In[21]:


plt.show()


# In[19]:


def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(tokenizer.tokenize(sent),2, pad_left=True,pad_right=True):
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

bigram_counts= bigram_model(str_sent)
print(bigram_counts['the']['jury'])


# In[20]:


def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(tokenizer.tokenize(sent),3, pad_left=True,pad_right=True):
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

trigram_counts= trigram_model(str_sent)
print(trigram_counts[('the','fulton')])


# In[21]:


test_sentences=['he lived a good life', 'the man was happy', 'the person was good' , 'the girl was sad', 'he won the war']

import numpy as np

test_unigram_arr=[]

print('Unigram test probabilities\n')
for elem in test_sentences:
    p_val = 1;
    for i in tokenizer.tokenize(elem):
        #print (unigram_counts[i])
        p_val = p_val * unigram_counts[i]
    #print (p_val)
    test_unigram_arr.append(p_val)
    print('The sequence : '+  elem +' => has unigram probablity of '+ str(p_val))


print('\nBigram test probabilities\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=1
    for w1,w2 in bigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=bigram_counts[w1][w2]
        except Exception as e:
                p_val=0
                break
    print('The sequence : '+ elem +' => has bigram probablity of '+ str(p_val))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test probabilities\n')
for elem in test_sentences:
    p_val=1
    for w1,w2,w3 in trigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=trigram_counts[(w1,w2)][w3]
        except Exception as e:
            p_val=0
            break
    print('The sequence : '+ elem +' => has trigram probablity of '+ str(p_val))
    
    test_trigram_arr.append(p_val)


# In[22]:


x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_by_value = sorted(x.items(), key=lambda kv: kv[1]


# In[23]:


import operator
d = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
sorted_d = sorted(d.items(), key=operator.itemgetter(1))


# In[ ]:


print(sorted_d)


# In[24]:


import operator
sorted_unigram_counts = list(reversed(sorted(unigram_counts.items(), key=operator.itemgetter(1))))


# In[34]:


print(sorted_unigram_counts[0:10])


# # Laplacian smoothing

# In[25]:



k=0.1
unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)+(k*N)

for word in unigram_counts:
    unigram_counts[word]=unigram_counts[word]+k
    unigram_counts[word]/=unigram_total
 
print(unigram_counts)


# In[26]:


def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(tokenizer.tokenize(sent),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))+(k*N)
        for w2 in model[w1]:
            model[w1][w2]=model[w1][w2]+k
            model[w1][w2]/=tot_count
     
    return model

bigram_counts= bigram_model(str_sent)
print(bigram_counts['the']['fulton'])


# In[27]:


def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(tokenizer.tokenize(sent),3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))+k*N
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]=model[(w1,w2)][w3]+k
            model[(w1,w2)][w3]/=tot_count
     
    return model

trigram_counts= trigram_model(str_sent)
print(trigram_counts[('the','fulton')])


# In[28]:


test_sentences=['he lived a good life', 'the man was happy', 'the person was good' , 'the girl was sad', 'he won the war']

import numpy as np

test_unigram_arr=[]

print('Unigram test probabilities\n')
for elem in test_sentences:
    p_val = 1;
    count = 0;
    for i in tokenizer.tokenize(elem):
        #print (unigram_counts[i])
        p_val = p_val * unigram_counts[i]
        count = count+1
    #print (p_val)
    #print (count)
    p_val = p_val ** ( (-1)/count)
    test_unigram_arr.append(p_val)
    print('The sequence : '+  elem +' => has unigram perplexity score of '+ str(p_val))


print('\nBigram test probabilities\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=1
    count =0;
    for w1,w2 in bigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=bigram_counts[w1][w2]
        except Exception as e:
                p_val*=1/N
                #break
        count = count+1
    #print (count)
    count = count -1
    if(p_val!=0):    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val = float('inf')
    print('The sequence : '+ elem +' => has bigram perplexity score of '+ str(p_val))
    
    test_bigram_arr.append(p_val)


test_trigram_arr=[]
print('\nTrigram test probabilities\n')
for elem in test_sentences:
    p_val=1
    count =0
    for w1,w2,w3 in trigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=trigram_counts[(w1,w2)][w3]
        except Exception as e:
            p_val*=1/N
            #break
        count = count+1
    #print (count)
    count = count-2
    if(p_val!=0):    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val = float('inf')
    print('The sequence : '+ elem +' => has trigram perplexity score of '+ str(p_val))
    
    test_trigram_arr.append(p_val)


# # Interpolation

# In[44]:


lamda=0.8
def bigram_model_interpolate(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(tokenizer.tokenize(sent),2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        for w2 in model[w1]:
            model[w1][w2]=model[w1][w2]
            model[w1][w2]/=tot_count
            print(model[w1][w2])
            print(unigram_counts[w1])
            model[w1][w2]=lamda*model[w1][w2]+(1-lamda)*unigram_counts[w1]
    return model

bigram_counts_interpolate= bigram_model_interpolate(str_sent)
print(bigram_counts['the']['fulton'])


# In[45]:


import math
for elem in test_sentences:
    p_val=1
    count =0;
    for w1,w2 in bigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=bigram_counts_interpolate[w1][w2]
        except Exception as e:
            p_val*=(1-lamda)*unigram_counts[w1]
        count = count+1
    #print (count)
    count = count -1
    if(p_val!=0): 
        p_val_log=math.log(p_val)
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has bigram log-likelihood score of '+ str(p_val_log))
    print('The sequence : '+ elem +' => has bigram perplexity score of '+ str(p_val))
    
    test_bigram_arr.append(p_val)


# In[ ]:


lines = [line.rstrip('\n') for line in open('testexamples.txt')]


# In[ ]:


test_sentences = [line.rstrip('\n') for line in open('testexamples.txt')]

