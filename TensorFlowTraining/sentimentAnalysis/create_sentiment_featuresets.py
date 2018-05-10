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
from nltk.tokenize import word_tokenize

#helps us in considering words like "run" and "ran" as the same thing, the verb "run" 
from nltk.stem import WordNetLemmatizer

import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()

#number of lines of the dataset
hm_lines = 10000000



#---------------------------------------------------------------
#pos -> positive dataset file name
#neg -> negative dataset file name
def create_lexicon(pos, neg):

	lexicon = []
	for fi in [pos, neg]:

		#opens the dataset file 
		with open(fi, 'r') as f:

			#contents contains all the lines from the file we just opened
			contents = f.readlines()
			for line in contents[:hm_lines]:
				
				#splits the line in words ans stores them in a list
				all_words = word_tokenize(line.lower())

				#adds the words in the lexicon
				lexicon += list(all_words)

	#?lemmatization -> remove prefixes and suffixes, so that words like "jump", "jumps" and "jumped" are treated as the same word 
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	
	#returns a list with each word and its frequency
	w_counts = Counter(lexicon)

	l2 = []

	#we are going do remove super-common words, like "the", "on", "of"....
	#we also remove super rare words
	for w in w_counts:

		if 1000 > w_counts[w] > 50:
			l2.append(w)


	print(len(l2))
	return l2


#---------------------------------------------------------------
def sample_handling (sample, lexicon, classification):

	featureset = []

	'''
	our goal here is to build a list like this:

	[

	[[0 1 0 1 1 0]   [1 0]] -> second, fourth and fifth words of the lexicon are present in the sentence. -> the sentence is positive and not negative

	[[0 0 1 1 1 0]   [0 1]] -> third, fourth and fifth words of the lexicon are present in the sentence. -> the sentence is not positive and is negative 

	]
	'''

	with open(sample, 'r') as f:
		contents = f.readlines()

		for line in contents[:hm_lines]:

			#separates a line of the sample in a list of words
			current_words = word_tokenize(line.lower())

			#lemmatizes the words of that line
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			
			#creates an array of zeros with the same length of our lexicon
			features = np.zeros(len(lexicon))

			#we are going to search each word of our sample in the lexicon
			for word in current_words:


				if word.lower() in lexicon:

					#if we find the word in the lexicon, we must retrieve the index in which that word is stored in the lexicon
					index_value = lexicon.index(word.lower())

					#we activate the coresponding feature in the analyzed sentence
					features[index_value] += 1

			features = list(features)
			featureset.append([features, classification])

	return featureset


#---------------------------------------------------------------
def create_feature_sets_and_labels(pos, neg, test_size=0.1):

	lexicon = create_lexicon(pos, neg)
	features = []
	features += sample_handling('pos.txt', lexicon, [1, 0])
	features += sample_handling('neg.txt', lexicon, [0, 1])	
	
	#shuffling the data for statistical reasons
	random.shuffle(features)

	features = np.array(features)

	testing_size = int(test_size*len(features))

	#======separates the data in test data and train data=====
	'''
	train_x = [
				[0 1 0 0 1 0 1 1]
				[0 0 1 0 1 0 0 1]
				[1 1 0 1 1 0 0 0]
				.
				.
				.
				except for the last 10 percent
			  ]
	'''
	train_x = list(features[:,0][:-testing_size]) 
	
	'''
	train_y = [
				[0 1]
				[1 0]
				[1 0] 
				.
				.
				.
				except for the last 10 percent
			  ]
	'''
	train_y = list(features[:,1][:-testing_size])


	#it's the same thing for test_x and test_y, except that it contains only the last 10 percent of the data
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:]) 

	return train_x, train_y, test_x, test_y




#---------------------------------------------------------------
if __name__ == '__main__':

	train_x, train_y, test_x, test_y = create_feature_sets_and_labels('pos.txt', 'neg.txt')

	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)


