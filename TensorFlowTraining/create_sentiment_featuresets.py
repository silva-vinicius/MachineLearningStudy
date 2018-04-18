#tutorial available at: https://pythonprogramming.net/using-our-own-data-tensorflow-deep-learning-tutorial/


#We'll create a lexicon - that is a list of all the unique words that are present on the dataset
'''
lexicon: [chair, table, spoon, television]

sentence: I pulled the chair up to the table


resulting array: [1 1 0 0]
Check which words in the lexicon are in the sentence: "chair" and "table"

index of "chair" in lexicon: 0 -> that's why the corresponding element in the resulting array turns 1
index of "table" in lexicon: 1 -> that's why the corresponding element in the resulting array turns 1

'''

import nltk
from nltk.tokenize import word_tokenizer

#helps us in considering words like "run" and "ran" as the same thing, the verb "run" 
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

#number of lines of the dataset
hm_lines = 10000000

def create_lexicon(pos, neg):

	lexicon = []
	for fi in [pos, neg]:

		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_counts = Counter(lexicon)

	l2 = []

	#we are going do remove super-common words, like "the", "on", "of"....
	#we also remove super rare words
	for w in w_counts:

		if 1000 > w_counts[w] > 50:
			l2.append(w)

	return l2


def sample_handling (sample, lexicon, classification):

	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()

		for l in contents[:hm_lines]:

			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))

			for

