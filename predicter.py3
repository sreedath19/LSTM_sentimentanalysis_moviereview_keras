import nltk
nltk.download('stopwords')
import nltk
nltk.download('wordnet')
import numpy as np
#from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers import Dense, Activation, Embedding
from keras import initializers
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import optimizers
from keras import models, layers
from keras import  backend as K
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing import sequence
		
import numpy as np
import re
import csv
stop_words=set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    
    return re.sub(strip_special_chars, "", string.lower())

print("Creating LSTM model")
e_init = initializers.RandomUniform(-0.01, 0.01, seed=1)
init = initializers.glorot_uniform(seed=1)
simple_adam = optimizers.Adam()
embed_vec_len = 32  
max_words = 25000

with open('idmatrix.csv', mode='r') as infile:
    reader = csv.reader(infile)
    Dict = {rows[0]:rows[1] for rows in reader}
def assignment(string):
	return Dict[string]

model = models.Sequential()
model.add(layers.embeddings.Embedding(input_dim=max_words,output_dim=embed_vec_len, embeddings_initializer=e_init,mask_zero=True))
model.add(layers.LSTM(units=100, kernel_initializer=init,dropout=0.2, recurrent_dropout=0.2))  # 100 memory
model.add(layers.Dense(units=1, kernel_initializer=init,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=simple_adam,metrics=['acc'])
print(model.summary())

model.load_weights("._Models_predicter.h5")

review = input('Enter the input text    :  ')



review = cleanSentences(review)

review = review.split()
sentance = []
for w in review:
	if w not in stop_words:
		sentance.append(w) 
final_sent =  [lemmatizer.lemmatize(i) for i in sentance]
words = final_sent
x = 0
for k in words:
	try:
		words[x] = int(assignment(words[x]))
	except KeyError:
		words[x] = 2
	x = x+1
print(words)
words = sequence.pad_sequences([words],truncating='pre', padding='pre', maxlen=80)
print(words)
review = words
print(review)

prediction = model.predict(review)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])



lstm_out = K.function([model.inputs[0], 
                        K.learning_phase()], 
                      [model.layers[2].output])
#pass in the input and set the the learning phase to 0
print(lstm_out([review, 0])) 


lstm_in = K.function([model.inputs[0], 
                        K.learning_phase()], 
                       [model.layers[1].output])
#pass in the input and set the the learning phase to 0
print(lstm_in([review, 0])) 




embed_out = K.function([model.inputs[0], 
                        K.learning_phase()], 
                       [model.layers[0].output])
# pass in the input and set the the learning phase to 0
print(embed_out([review, 0])) 

