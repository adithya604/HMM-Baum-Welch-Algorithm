import numpy
import json
import os
from nltk.corpus import brown
from collections import OrderedDict
from random import *
import numpy as np

# model = {
# 	'A' : {
# 		'CP' : {'CP' : 0.7, 'IP' : 0.3},
# 		'IP' : {'CP' : 0.5, 'IP' : 0.5}
# 	},
# 	'B' : {
# 		'CP' : {'cola' : 0.6, 'ice_t' : 0.1, 'lem' : 0.3},
# 		'IP' : {'cola' : 0.1, 'ice_t' : 0.7, 'lem' : 0.2},
# 	},
# 	'pi' : {
# 		'CP' : 1.0,
# 		'IP' : 0.0
# 	}
# }
observation = ['lem', 'ice_t', 'cola']
#observation = brown.sents()[0]
observation = []
for w in brown.sents()[0]:
	observation.append(w.lower().encode('ascii', 'ignore'))

def forwardProcedure(model, observation):
	#Initialize alphas
	states = model['pi'].keys()
	alphas = {}
	for state in states:
		alphas[state] = []

	## Initialization Step
	for state in states:
		alphas[state].append(model['pi'][state])

	## Induction Step
	for t in range(len(observation)):
		for sj in states:
			summ = sum((alphas[si][t]*model['A'][si][sj]*model['B'][si][observation[t]]) \
													for si in states)
			alphas[sj].append(summ)
		t += 1

	## Last Step ie Total =  P(O|model)
	prob = sum((alphas[state][t]) for state in states)

	return alphas, prob

def backwardProcedure(model, observation):
	#Initialize betas
	states = model['pi'].keys()
	betas = {}
	for state in states:
		betas[state] = []

	## Initialization Step
	for state in states:
		betas[state].append(1.0)

	## Induction Step
	observation.reverse() # taking obs seq in reverse order
	for t in range(len(observation)):
		for si in states:
			summ = sum((betas[sj][t]*model['A'][si][sj]*model['B'][si][observation[t]]) \
													for sj in states)
			betas[si].append(summ)
		t += 1

	## Reverse obs seq back and the list to match 't' of alphas with 't' of betas
	observation.reverse()
	for state in states:
		betas[state].reverse()

	## Last Step ie Total =  P(O|model)
	prob = sum((betas[state][0]*model['pi'][state]) for state in states)

	return betas, prob

def baumWelshTraining(model, observation, alphas, betas):
	states = model['pi'].keys()

	zeta = [{} for t in range(len(observation))]
	gammas = [{} for t in range(len(observation))]

	for t in range(len(observation)):
		for si in states:
			zeta[t][si] = {}
			summ = 0
			for sj in states:
				zeta[t][si][sj] = (alphas[si][t] * model['A'][si][sj] * model['B'][si][observation[t]] * \
									betas[sj][t+1])/float(fwdProb)	
				#print t, si, sj, observation[t], alphas[si][t], model['A'][si][sj], model['B'][si][observation[t]], betas[sj][t+1], zeta[t][si][sj]
				summ += zeta[t][si][sj]	 # for gammas
			gammas[t][si] = summ
			if t == 0:
				model['pi'][si] = gammas[t][si]
	# print "ZETA --- "
	# print zeta

	# print "GAMMAS --- "
	# print gammas

	# print "PIII --- "
	# print model['pi']
	for si in states:
		for sj in states:
			numer = sum([zeta[t][si][sj] for t in range(len(observation))])
			denom = sum([gammas[t][si] for t in range(len(observation))])
			model['A'][si][sj] = numer/denom
		summ = sum(model['A'][si].values())
		for sj in states:
			model['A'][si][sj] /= summ

	# print "A --- "
	# print model['A']
	keyss = model['B'][model['B'].keys()[0]].keys()

	for si in states:
		for obs in keyss:
			numer = sum([gammas[t][si] for t in range(len(observation)) if observation[t] == obs])
			denom = sum([gammas[t][si] for t in range(len(observation))])
			model['B'][si][obs] = numer/denom
		summ = sum(model['B'][si].values())
		for obs in observation:
			model['B'][si][obs] /= summ
	# print "B --- "
	# print model['B']


def getVocabulary(corpus, noSentences):
	voc = []
	for i in range(noSentences):
		voc += corpus.sents()[i]
	print "Leng ", len(voc)
	for i in range(len(voc)):
		voc[i] = voc[i].lower().encode('ascii', 'ignore')
	print "Finished in gettting vocabulary "
	return list(set(voc))

def getDataFromCorpus(corpus, no_sentences = 100, no_states = 10):
	print len(corpus.sents()), "Hello"
	print "No of sentences ", no_sentences
	vocabulary = getVocabulary(corpus, no_sentences) ## needed for emission matrix

	leng = len(vocabulary)
	print "Trimmed vocabulary ", leng

	print "Generating random A, B, pi ie model"
	A = {} ## transition matrix
	for t1 in range(no_states):
		A['s'+str(t1)] = {}
		for t2 in range(no_states):
			A['s'+str(t1)]['s'+str(t2)] = random()
	for si in A:
		summ = sum(A[si].values())
		for sj in A[si].keys():
			A[si][sj] /= summ
	B = {} ## Emission matrix
	for t1 in range(no_states):
		B['s'+str(t1)] = {}
		for vocab in vocabulary:
			B['s'+str(t1)][vocab] = random()
	for si in B:
		summ = sum(B[si].values())
		for sj in B[si].keys():
			B[si][sj] /= summ
	pi = {}
	for t in range(no_states):
		pi['s'+str(t)] = random()
	summ = sum(pi.values())
	for t in range(no_states):
		pi['s'+str(t)] /= summ
	model = {}
	model['A']  = A
	model['B']  = B
	model['pi'] = pi
	print "Model created...!"
	return model, vocabulary

	
model, emmissionKeys = getDataFromCorpus(brown, no_sentences=1000)
#print "\n Model is ", model

print "\n Observation seq is ", observation

prevProb = 1
no_iterations = 100
i = 1
while(1):
	print "Iteration ", i
	i += 1

	alphas, fwdProb = forwardProcedure(model, observation)
	#print "\n After Training, Forward Procedure :"
	print "Forward Prob -- ", fwdProb
	betas, backProb = backwardProcedure(model, observation)
	#print "\n After Backward Procedure :"
	print "Backward Prob ", backProb

	baumWelshTraining(model, observation, alphas, betas)

	if abs(prevProb-fwdProb) == 0:
		break;
	prevProb = fwdProb