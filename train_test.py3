import numpy as np
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


print("Creating LSTM model")
e_init = initializers.RandomUniform(-0.01, 0.01, seed=1)
init = initializers.glorot_uniform(seed=1)
simple_adam = optimizers.Adam()
embed_vec_len = 32  
max_words = 25000


model = models.Sequential()
model.add(layers.embeddings.Embedding(input_dim=max_words,output_dim=embed_vec_len, embeddings_initializer=e_init,mask_zero=True))
model.add(layers.LSTM(units=100, kernel_initializer=init,dropout=0.2, recurrent_dropout=0.2))  # 100 memory
model.add(layers.Dense(units=1, kernel_initializer=init,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=simple_adam,metrics=['acc'])
print(model.summary())

train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

test_x = np.load('test_x.npy') 
test_y = np.load('test_y.npy')

bat_size = 32
max_epochs = 3
print("\nStarting training ")
model.fit(train_x, train_y, validation_data = (test_x,test_y), epochs=max_epochs,batch_size=bat_size, shuffle=True, verbose=1) 
print("Training complete \n")


loss_acc = model.evaluate(test_x, test_y, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " %(loss_acc[0], loss_acc[1]*100))

print("Saving model to disk \n")
mp = ".\\Models\\predicter.h5"
model.save(mp)


