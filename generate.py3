import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from os import listdir
from os.path import isfile, join
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
import numpy
from numpy import array
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

stop_words=set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


from keras.preprocessing import sequence
from sklearn.feature_extraction.text import TfidfVectorizer		
import numpy as np
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
vec = TfidfVectorizer()
import csv


with open('idmatrix.csv', mode='r') as infile:
    reader = csv.reader(infile)
    Dict = {rows[0]:rows[1] for rows in reader}
def assignment(string):
	return Dict[string]


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    
    return re.sub(strip_special_chars, "", string.lower())
ids = np.zeros((25000,80))
i = 0 
#the_data_set = ''
for pf in positiveFiles:
	with open(pf, "r") as f:
		line=f.read()
		cleanedLine = cleanSentences(line)
		words = cleanedLine.split()
		filtered_sentance = []
		for w in words:
			if w not in stop_words:
				filtered_sentance.append(w) 

		final_sent =  [lemmatizer.lemmatize(i) for i in filtered_sentance]
		words = final_sent
		#the_data_set =the_data_set + line
		x = 0
		for k in words:
			try:
				words[x] = int(assignment(words[x]))
			except KeyError:
				words[x] = 2
			x = x+1
		words = array(words)
		print(words)
		words.flatten()
		words = sequence.pad_sequences([words],truncating='pre', padding='pre', maxlen=80)
		j=0
		print(words.shape)
		while j<80:
			print(words[0,j])
			ids[i,j] = words[0,j]
			j=j+1
		i = i+1

		
for nf in negativeFiles:
	with open(nf, "r") as f:
		line=f.read()
		cleanedLine = cleanSentences(line)
		words = cleanedLine.split()
		filtered_sentance = []
		for w in words:
			if w not in stop_words:
				filtered_sentance.append(w) 

		final_sent =  [lemmatizer.lemmatize(i) for i in filtered_sentance]
		words = final_sent
		#the_data_set =the_data_set + line
		x = 0
		for k in words:
			try:
				words[x] = int(assignment(words[x]))
			except KeyError:
				words[x] = 2
			x = x+1
		words = array(words)
		print(words)
		words.flatten()
		words = sequence.pad_sequences([words],truncating='pre', padding='pre', maxlen=80)
		j=0
		print(words)
		while j<80:
			ids[i,j] = words[0,j]
			j=j+1
		i = i+1
print(i)
#the_data_set = cleanSentences(the_data_set)
#words = the_data_set.split()
#filtered_sentance = []
#for w in words:
#	if w not in stop_words:
#		filtered_sentance.append(w) 
#final_sent =  [lemmatizer.lemmatize(i) for i in filtered_sentance]
#words = final_sent

#Counter = Counter(words)
#most_occur = Counter.most_common(20000) 
#j=3
#for i in range (0,20000):
#	most_occur[i] = list(most_occur[i])
#	most_occur[i][1] = j
#	most_occur[i] = tuple(most_occur[i])
#	j=j+1
#Dict = {}	
#for i in most_occur:
#	Dict[i[0]] = i[1]
#
#import csv
#with open('test.csv', 'w') as f:
#    for key in Dict.keys():
#       f.write("%s,%s\n"%(key,Dict[key]))
np.save('idmatmagic', ids)
np.savetxt('idmatmagic.csv', ids, "%d", "")



