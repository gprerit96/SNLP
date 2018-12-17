import numpy as np
import nltk, pylab
from nltk.corpus import brown
from nltk import bigrams, ngrams, trigrams
from collections import Counter
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import operator

#Training
num_sent=40000
corpus=brown.sents()[0:num_sent]
sentences=[]
for ls in corpus: #Remove special characters and Convert to lowercase 
	sentences.append([w.lower() for w in ls if w.isalpha()])
print('\n----------Assignment Task1----------\n')
## Build unigram dictionary

def unigram_model(sentences):
	unigrams=[]
	for ls in sentences:
	        unigrams.extend(ls)

	unigram_counts=Counter(unigrams)
	unigram_total=len(unigrams)

	for word in unigram_counts:
		unigram_counts[word]/=unigram_total

	return unigram_counts

unigram_counts=unigram_model(sentences)
Unigram_Counts=unigram_counts ##To be used in 4th part

N=len(unigram_counts)
print ("Top 10 Unigrams are")	
counts=defaultdict(int)
for ls in sentences:
	for w1 in ngrams(ls,1, pad_left=False,pad_right=False):
		counts[w1]+=1
unigrams_freq = Counter(counts)
print(unigrams_freq.most_common(10))

sorted_unigram = list(reversed(sorted(unigrams_freq.items(), key=operator.itemgetter(1))))
N=len(sorted_unigram)
i=1
r=[]
while(i<=N):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1    
f=[math.log(y) for (x,y) in sorted_unigram]
plt.plot(f,r)
plt.xlabel('log(i)')
plt.ylabel('log(probs)')
plt.title('Zipfs law plot : Unigram model')
plt.show()

## Build bigram dictionary
def bigram_model(sentences):
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
			model[w1][w2]/=tot_count
	return model

bigram_counts=bigram_model(sentences)

# print(bigram_counts.most_common(10))
# print(bigram_counts)
	
print ("Top 10 Bigrams are")	
counts=defaultdict(int)
for ls in sentences:
	for w1,w2 in ngrams(ls,2, pad_left=False,pad_right=False):
		counts[(w1,w2)]+=1
bigrams_freq = Counter(counts)
print(bigrams_freq.most_common(10))

sorted_bigram = list(reversed(sorted(bigrams_freq.items(), key=operator.itemgetter(1))))
N=len(sorted_bigram)
i=1
r=[]
while(i<=N):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1    
f=[math.log(y) for (x,y) in sorted_bigram]
plt.plot(f,r)
plt.xlabel('log(Index)')
plt.ylabel('log(Frequency)')
plt.title('Zipfs law plot : Bigram model')
plt.show()

## Build trigram dictionary
def trigram_model(sentences):
	model={}
	for ls in sentences:
		for w1,w2,w3 in ngrams(ls,3, pad_left=True,pad_right=True):
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

#print(trigram_counts)
print ("Top 10 Trigrams are")	
counts=defaultdict(int)
for ls in sentences:
	for w1,w2,w3 in ngrams(ls,3, pad_left=False,pad_right=False):
		counts[(w1,w2,w3)]+=1
trigrams_freq = Counter(counts)
print(trigrams_freq.most_common(10))
sorted_trigram = list(reversed(sorted(trigrams_freq.items(), key=operator.itemgetter(1))))
N=len(sorted_trigram)
i=1
r=[]
while(i<=N):
    log_i=math.log(i)
    r.append(log_i)
    i=i+1    
f=[math.log(y) for (x,y) in sorted_trigram]
plt.plot(f,r)
plt.xlabel('log(Index)')
plt.ylabel('log(Frequency)')
plt.title('Zipfs law plot : Trigram model')
plt.show()


#Testing
# test_sentences=["he lived a good life","the man was happy","the person was good","the girl was sad","he won the war"]

f=open("test_examples.txt", "r")
test_sentences=f.read().splitlines()


test_unigram_arr=[]
print('\n')
print("Unigram test probabilities\n")
for elem in test_sentences:
	p_val=1
	count=0
	for i in elem.split():
		p_val*=unigram_counts[i]
		count=count+1
	per=p_val**(-(1.0)/count)
	loglike=-math.log(p_val,2)
	#print(count)
	test_unigram_arr.append(p_val)
	print('The sequence ("'+elem+'") has unigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))


print('\nBigram test probabilities\n')
test_bigram_arr=[]
for elem in test_sentences:
	p_val=1
	count=0
	for w1,w2 in bigrams(elem.split(),pad_left=True,pad_right=True):
		if w1 in bigram_counts: 
			# print(w1,w2)
			if w2 in bigram_counts[w1]:
				p_val*=bigram_counts[w1][w2]
			else:
				p_val=0
				break
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
	test_bigram_arr.append(p_val)
		

test_trigram_arr=[]
print('\nTrigram test probabilities \n')
for elem in test_sentences:
	p_val=1
	count=0
	for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
		try:
			p_val*=trigram_counts[(w1,w2)][w3]
		except Exception as e:
			#print('Exception has occurred')
			p_val=0
			break
		count=count+1
	# print(count)
	count=count-2
	if p_val==0 :
		per=float('INF')
		loglike=float('INF')
	else : 
		per=p_val ** ( (-1)/count)
		loglike=-math.log(p_val,2)
	print('The sequence ("'+ elem +'") has trigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))
	test_trigram_arr.append(p_val)



print('\n----------Assignment Task2: Laplacian/Additive Smoothing----------\n')
K=0.01
print('Smoothing Parameter k='+str(K))
def unigram_model_additive_smoothing(sentences,k):
	unigrams=[]
	for ls in sentences:
	        unigrams.extend(ls)

	unigram_counts=Counter(unigrams)
	unigram_total=len(unigrams)+k*N

	for word in unigram_counts:
		unigram_counts[word]=(unigram_counts[word]+k)/unigram_total
	return unigram_counts
unigram_counts=unigram_model_additive_smoothing(sentences,K)

def bigram_model_additive_smoothing(sentences,k):
	model={}
	for ls in sentences:
		for w1,w2 in ngrams(ls,2, pad_left=True,pad_right=True):
			if w1 not in model:
				model[w1]={}
			if w2 not in model[w1]:
				model[w1][w2]=0
			model[w1][w2]+=1

	for w1 in model:	
		tot_count=float(sum(model[w1].values()))+k*N
		for w2 in model[w1]:
			model[w1][w2]=(model[w1][w2]+k)/tot_count
	return model

bigram_counts=bigram_model_additive_smoothing(sentences,K)


def trigram_model_additive_smoothing(sentences,k):
	model={}
	for ls in sentences:
		for w1,w2,w3 in ngrams(ls,3, pad_left=True,pad_right=True):
			if (w1,w2) not in model:
				model[(w1,w2)]={}
			if w3 not in model[(w1,w2)]:
				model[(w1,w2)][w3]=0
			model[(w1,w2)][w3]+=1

	for (w1,w2) in model:
		tot_count=float(sum(model[(w1,w2)].values()))+k*N
		for w3 in model[(w1,w2)]:
			model[(w1,w2)][w3]=(model[(w1,w2)][w3]+k)/tot_count
	return model

trigram_counts= trigram_model_additive_smoothing(sentences,K)


print('\n')
print("Unigram test probabilities\n")
for elem in test_sentences:
	p_val=1
	count=0
	for i in elem.split():
		p_val*=unigram_counts[i]
		count=count+1
	per=p_val**(-(1.0)/count)
	loglike=-math.log(p_val,2)
	#print(count)
	print('The sequence ("'+elem+'") has unigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))


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
				p_val*=1/N
			count=count+1
	# print(count)
	count=count-1
	if p_val==0 :
		per=float('INF')
		loglike=float('INF')
	else : 
		per=p_val**((-1.0)/count)
		loglike=-math.log(p_val,2)
	print('The sequence ("'+ elem +'") has bigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))
	test_bigram_arr.append(p_val)
		

print('\nTrigram test probabilities \n')
for elem in test_sentences:
	p_val=1
	count=0
	for w1,w2,w3 in trigrams(elem.split(),pad_left=True,pad_right=True):
		try:
			p_val*=trigram_counts[(w1,w2)][w3]
		except Exception as e:
			#print('Exception has occurred')
			p_val*=1/N
		count=count+1
	# print(count)
	count=count-2
	if p_val==0 :
		per=float('INF')
		loglike=float('INF')
	else : 
		per=p_val**((-1.0)/count)
		loglike=-math.log(p_val,2)
	print('The sequence ("'+ elem +'") has trigram probablity of '+ str(p_val)+' log-likelihood score of '+ str(loglike)+' and perplexity score of '+str(per))


print('\n----------Assignment Task3: Simple Good Turing Smoothing----------\n')





print('\n----------Assignment Task4: Interpolation Method----------\n')
Lamda=0.2
print('Interpolation Parameter Lamda='+str(Lamda))

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
			model[w1][w2]=Lamda*model[w1][w2]+(1-Lamda)*Unigram_Counts[w1]
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
				p_val*=(1-Lamda)*Unigram_Counts[w1]
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





# x_axis=[i for i in range(1,4)]
# y_axis=[np.mean(test_unigram_arr), np.mean(test_bigram_arr), np.mean(test_trigram_arr)]
# plt.scatter(x_axis,y_axis)
# plt.show()

# def zipf_plot(unigram_counts):
#     fdist = nltk.FreqDist(unigram_counts)
#     pylab.plot(
#             range(1, fdist.B() + 1),      # x-axis: word rank
#             fdist.values()                # y-axis: word count
#             )   
#     pylab.xscale('log')
#     pylab.yscale('log')
#     pylab.show()

# zipf_plot(unigram_counts)
# # zipf_plot(bigram_counts)
# # zipf_plot(trigram_counts)
