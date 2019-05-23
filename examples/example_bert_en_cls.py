from keras.layers import *
from keras.models import *

from imdb import *
max_len = 150

def convert_1str(df):
	text = df['sentence'].tolist()
	text = [' '.join(t.split()[:max_len]) for t in text]
	label = df['polarity'].tolist()
	return text, label

(train_text, train_label), (test_text, test_label) = map(convert_1str, (train_df, test_df))

import sys
sys.path.append('../')
import bert
bert.set_language('en')

train_inputs, test_inputs = map(lambda x:bert.convert_sentences(x, max_len), [train_text, test_text])

bert_inputs = bert.get_bert_inputs(max_len)
bert_output = bert.BERTLayer(n_fine_tune_vars=3)(bert_inputs)
x = Dense(256, activation='relu')(bert_output)
x = Dense(1, activation='sigmoid')(x)
model = Model(inputs=bert_inputs, outputs=x)

epochs = 2
batch_size = 32
total_steps = epochs*train_inputs[0].shape[0]//batch_size
lr_scheduler, opt_lr = bert.get_suggested_scheduler(init_lr=1e-3, total_steps=total_steps)
optimizer = bert.get_suggested_optimizer(opt_lr)
	
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()
	
model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size, 
		  validation_data=(test_inputs, test_label), callbacks=[lr_scheduler])

'''
Epoch 1/2
25000/25000 [==============================] - 405s 16ms/step - loss: 0.3651 - acc: 0.8408 - val_loss: 0.3603 - val_acc: 0.8429
Epoch 2/2
25000/25000 [==============================] - 403s 16ms/step - loss: 0.2877 - acc: 0.8786 - val_loss: 0.2897 - val_acc: 0.8764
'''