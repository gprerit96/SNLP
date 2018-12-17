import numpy as np
import re,math
import nltk
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams
from collections import Counter
import matplotlib.pyplot as plt 
import operator

test_sentences = [line.rstrip('\n') for line in open('text_examples.txt')]
raw_sent = brown.sents()[:40000]

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

######## Pre-Processing of raw sentences ########
sentences = []
for sent in raw_sent:
	sentence = []
	for word in sent:
		word = re.sub(r'[^a-zA-Z ]',r'',word)
		if(len(word)>0):
			sentence.append(word.lower())
	if(len(sentence)>0):
		sentences.append(sentence)

##### Unigram Model #####
unigrams=[]

for sent in sentences:
    unigrams.extend(sent)

unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)

for word in unigram_counts:
    unigram_counts[word]/=unigram_total
 
#print(list(unigram_counts.items())[:10])
sorted_unigram = list(reversed(sorted(unigram_counts.items(), key=operator.itemgetter(1))))
print ("The top 10 unigrams with their probabilities are: ",sorted_unigram[:10])

###### Zipf's Law plot for Unigram Model #######
N1=len(sorted_unigram)
i=1
r=[]
while(i<=N1):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1
    
f=[math.log(y) for (x,y) in sorted_unigram]

plt.plot(f,r)
plt.xlabel('log(i)')
plt.ylabel('log(probs)')
plt.title('Zipfs law plot : Unigram model')
plt.show()


###### Bigram Model ######
def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
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

def bi_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]=0
            model[(w1,w2)]+=1
    tot_count=float(sum(model.values()))
    for (w1,w2) in model: 
        model[(w1,w2)]/=tot_count
     
    return model

bigram_counts= bigram_model(sentences)
bi_counts = bi_model(sentences)
sorted_bigram = list(reversed(sorted(bi_counts.items(), key=operator.itemgetter(1))))
print ("The top 10 bigrams with their probabilities are: ",sorted_bigram[:10])

######## Zipf's law plot for Bigram Model #######
N2=len(sorted_bigram)
i=1
r=[]
while(i<=N2):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1
    
f=[math.log(y) for (x,y) in sorted_bigram]

plt.plot(f,r)
plt.xlabel('log(i)')
plt.ylabel('log(probs)')
plt.title('Zipfs law plot : Bigram model')
plt.show()

#print(list(bigram_counts.items())[:10])

##### Trigram Model ######
def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
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

def tri_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
            if (w1,w2,w3) not in model:
                model[(w1,w2,w3)]=0
            model[(w1,w2,w3)]+=1
    tot_count=float(sum(model.values()))
    for (w1,w2,w3) in model: 
        model[(w1,w2,w3)]/=tot_count     
    return model

trigram_counts= trigram_model(sentences)
#print(list(trigram_counts.items())[:10])
tri_counts = tri_model(sentences)
sorted_trigram = list(reversed(sorted(tri_counts.items(), key=operator.itemgetter(1))))
print ("The top 10 trigrams with their probabilities are: ",sorted_trigram[:10])

######## Zipf's law plot for Trigram Model #######
N3=len(sorted_trigram)
i=1
r=[]
while(i<=N3):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1
    
f=[math.log(y) for (x,y) in sorted_trigram]

plt.plot(f,r)
plt.xlabel('log(i)')
plt.ylabel('log(probs)')
plt.title('Zipfs law plot : Trigram model')
plt.show()

#### Find the probabilities ######

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
    p_val_log = -math.log(p_val)
    p_val = p_val ** ( (-1)/count)
    test_unigram_arr.append(p_val)
    print('The sequence : '+ elem +' => has unigram log-likelihood score of '+ str(p_val_log))
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
                p_val*=1/N1
                #break
        count = count+1
    #print (count)
    count = count -1
    if(p_val!=0):
        p_val_log = -math.log(p_val)    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has bigram log-likelihood score of '+ str(p_val_log))
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
            p_val*=1/N1
            #break
        count = count+1
    #print (count)
    count = count-2
    if(p_val!=0): 
        p_val_log = -math.log(p_val)   
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has trigram log-likelihood score of '+ str(p_val_log))
    print('The sequence : '+ elem +' => has trigram perplexity score of '+ str(p_val))
    
    test_trigram_arr.append(p_val)


###### Laplacian Smoothing #######
print ("Laplacian Smoothing")
k=0.1

unigram_counts=Counter(unigrams)
unigram_total=len(unigrams)+(k*N1)

for word in unigram_counts:
    unigram_counts[word]=unigram_counts[word]+k
    unigram_counts[word]/=unigram_total
 
#print(unigram_counts)


def bigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]=0
            model[w1][w2]+=1
    for w1 in model:
        tot_count=float(sum(model[w1].values()))+(k*N1)
        for w2 in model[w1]:
            model[w1][w2]=model[w1][w2]+k
            model[w1][w2]/=tot_count
     
    return model

bigram_counts= bigram_model(sentences)
print(bigram_counts['the']['fulton'])


def trigram_model(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=0
            model[(w1,w2)][w3]+=1
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))+k*N1
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]=model[(w1,w2)][w3]+k
            model[(w1,w2)][w3]/=tot_count
     
    return model

trigram_counts= trigram_model(sentences)
print(trigram_counts[('the','fulton')])

######### Testing the sample sentences ###########
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
    p_val_log = -math.log(p_val)
    p_val = p_val ** ( (-1)/count)
    test_unigram_arr.append(p_val)
    print('The sequence : '+ elem +' => has unigram log-likelihood score of '+ str(p_val_log))
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
                p_val*=1/N1
                #break
        count = count+1
    #print (count)
    count = count -1
    if(p_val!=0):
        p_val_log = -math.log(p_val)    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has bigram log-likelihood score of '+ str(p_val_log))
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
            p_val*=1/N1
            #break
        count = count+1
    #print (count)
    count = count-2
    if(p_val!=0): 
        p_val_log = -math.log(p_val)   
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has trigram log-likelihood score of '+ str(p_val_log))
    print('The sequence : '+ elem +' => has trigram perplexity score of '+ str(p_val))
    
    test_trigram_arr.append(p_val)

#Defining the good turing method
print("Smoothing using Good Turing method")
N = N1
def bigram_model_gt(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]=0
            model[(w1,w2)]+=1
            
     
    return model

bigram_counts_gt= bigram_model_gt(sentences)
print (bigram_counts_gt[('the','fulton')])

ma = 0
freq = []
prob = []
tot_bi = 0
for word in bigram_counts_gt:
    
    if (bigram_counts_gt[word]>ma):
        ma = bigram_counts_gt[word]
    tot_bi = tot_bi + 1
    
i=0
while(i<=ma):
    freq.append(0)
    prob.append(0)
    i=i+1
    
for word in bigram_counts_gt:
    freq[bigram_counts_gt[word]] = freq[bigram_counts_gt[word]] + 1
    
print (tot_bi)
freq[0]= N*N - tot_bi
#print (freq)

i=0
tot_p = 0

while(i<len(prob)):
    if(freq[i]):
        j=i+1
        while(j<len(prob)):
            if(freq[j]):
                break
            j=j+1
        if(j<len(freq)):    
            prob[i] = (j*freq[j])/freq[i]
    else:
        prob[i] = 0
    tot_p = tot_p + prob[i]
    i=i+1
    
    
#print (prob)
print (tot_p)

i=0
while(i<len(prob)):
    prob[i] = prob[i]/tot_p
    i=i+1
#print (prob)

def bigram_model_gt_prob(sentences):
    model={}
    for sent in sentences:
         for w1,w2 in ngrams(sent,2, pad_left=True,pad_right=True):
            if w1 not in model:
                model[w1]={}
            if w2 not in model[w1]:
                model[w1][w2]= prob[bigram_counts_gt[(w1,w2)]]
            #model[w1][w2]+=1
    #print(N)
    for w1 in model:
        tot_count=float(sum(model[w1].values()))
        tot_count = tot_count + (N - len(model[w1]))*prob[0]
        for w2 in model[w1]:
            model[w1][w2]/=tot_count
        model[w1]['.'] = prob[0]/tot_count
    return model

bigram_probs_gt= bigram_model_gt_prob(sentences)

print (float(sum(bigram_probs_gt['the'].values())))
print (len(bigram_probs_gt['the']))



def trigram_model_gt(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
            if (w1,w2,w3) not in model:
                model[(w1,w2,w3)]=0
            model[(w1,w2,w3)]+=1

     
    return model

trigram_counts_gt= trigram_model_gt(sentences)
print (trigram_counts_gt[('the','fulton','county')])

ma = 0
freq_t = []
prob_t = []
tot_ti = 0
for word in trigram_counts_gt:
    
    if (trigram_counts_gt[word]>ma):
        ma = trigram_counts_gt[word]
    tot_ti = tot_ti + 1
    
i=0
while(i<=ma):
    freq_t.append(0)
    prob_t.append(0)
    i=i+1
    
for word in trigram_counts_gt:
    freq_t[trigram_counts_gt[word]] = freq_t[trigram_counts_gt[word]] + 1
    
print (tot_ti)
freq_t[0]= N*N*N - tot_ti

i=0
tot_p_t = 0

while(i<len(prob_t)):
    if(freq_t[i]):
        j=i+1
        while(j<len(prob_t)):
            if(freq_t[j]):
                break
            j=j+1
        if(j<len(freq_t)):    
            prob_t[i] = (j*freq_t[j])/freq_t[i]
    else:
        prob_t[i] = 0
    tot_p_t = tot_p_t + prob_t[i]
    i=i+1
    

print (tot_p_t)

i=0
while(i<len(prob_t)):
    prob_t[i] = prob_t[i]/tot_p_t
    i=i+1

def trigram_model_gt_prob(sentences):
    model={}
    for sent in sentences:
         for w1,w2,w3 in ngrams(sent,3, pad_left=True,pad_right=True):
            if (w1,w2) not in model:
                model[(w1,w2)]={}
            if w3 not in model[(w1,w2)]:
                model[(w1,w2)][w3]=prob_t[trigram_counts_gt[(w1,w2,w3)]]
    for (w1,w2) in model:
        tot_count=float(sum(model[(w1,w2)].values()))
        tot_count = tot_count + (N - len(model[(w1,w2)]))*prob_t[0]
        for w3 in model[(w1,w2)]:
            model[(w1,w2)][w3]/=tot_count
        model[(w1,w2)]['.'] = prob_t[0]/tot_count
    return model

trigram_probs_gt= trigram_model_gt_prob(sentences)
print(trigram_probs_gt[('the','fulton')])

print('\nBigram test probabilities\n')

test_bigram_arr=[]

for elem in test_sentences:
    p_val=1
    count =0;
    for w1,w2 in bigrams(tokenizer.tokenize(elem),pad_left=True,pad_right=True):
        try:
            p_val*=bigram_probs_gt[w1][w2]
        except Exception as e:
                p_val=p_val*bigram_probs_gt[w1]['.']
                #break
        count = count+1
    #print (count)
    count = count -1
    if(p_val!=0):
        p_val_log=-math.log(p_val)    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has log-likelihood score of '+ str(p_val_log))
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
            try:
                p_val*=trigram_counts[(w1,w2)]['.']
            except Exception as e:
                p_val=p_val*prob_t[0]
            #break
        count = count+1
    #print (count)
    count = count-2
    if(p_val!=0):
        p_val_log=-math.log(p_val)    
        p_val = p_val ** ( (-1)/count)
    else:
        p_val_log = float('inf')
        p_val = float('inf')
    print('The sequence : '+ elem +' => has log-likelihood score of '+ str(p_val_log))
    print('The sequence : '+ elem +' => has bigram perplexity score of '+ str(p_val))
    
    test_trigram_arr.append(p_val)

######### Interpolation ########

print ("Results for Interpolation")

for Lamda in [0.2,0.5,0.8]:
    print('Interpolation Parameter Lamda='+str(Lamda))
    K = k
    def bigram_model_interpolate(sentences,k):
        model={}
        for ls in sentences:
            for w1,w2 in ngrams(ls,2, pad_left=True,pad_right=True):
                if w1 not in model:
                    model[w1]={}
                if w2 not in model[w1]:
                    model[w1][w2]=0
                model[w1][w2]+=1

        for w1 in model:    
            tot_count=float(sum(model[w1].values()))
            for w2 in model[w1]:
                model[w1][w2]=model[w1][w2]/tot_count
                model[w1][w2]=Lamda*model[w1][w2]+(1-Lamda)*unigram_counts[w1]
        return model

    bigram_counts=bigram_model_interpolate(sentences,K)

    print('\nBigram test probabilities\n')

    for elem in test_sentences:
        p_val=1
        count=0
        for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
            if w1 in bigram_counts: 
                # print(w1,w2)
                if w2 in bigram_counts[w1]:
                    p_val*=bigram_counts[w1][w2]
                else:
                    p_val*=(1-Lamda)*unigram_counts[w1]
                count=count+1
        # print(count)
        count=count-1
        if p_val==0 :
            per=float('INF')
            loglike=float('INF')
        else : 
            per=p_val ** ( (-1)/count)
            loglike=-math.log(p_val,2)
        print('The sequence ("'+ elem +'") has bigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))



