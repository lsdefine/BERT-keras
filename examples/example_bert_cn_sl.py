import sys, ljqpy
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras_contrib.layers import CRF
from keras.preprocessing import sequence

max_len = 100

def LoadCoNLLFormat(fn='../dataset/conll2003/en/train.txt', tag_column=-1, has_headline=False):
	datax, datay = [], []
	tempx, tempy = [], []
	with open(fn, encoding='utf-8') as fin:
		for lln in fin:
			lln = lln.strip()
			if has_headline or lln.startswith('-DOCSTART-'):
				has_headline = False; continue
			if lln == '':
				if len(tempx) >= 1:
					datax.append(tempx); datay.append(tempy)
				tempx, tempy = [], []
			else:
				items = lln.split()
				tempx.append(items[0])
				tempy.append(items[tag_column])
	if len(tempx) >= 1:
		datax.append(tempx); datay.append(tempy)
	return datax, datay

datadir = 'chsner_char-level'
xys = [LoadCoNLLFormat(os.path.join(datadir, '%s.txt') % tp) for tp in ['train', 'test']]

id2y = {}
for yy in xys[0][1]:
	for y in yy: id2y[y] = id2y.get(y, 0) + 1
id2y = [x[0] for x in ljqpy.FreqDict2List(id2y)]
y2id = {v:k for k,v in enumerate(id2y)}

def convert_data(df):
	text = [' '.join(t[:max_len]) for t in df[0]]
	label = [[0]+[y2id.get(x, 0) for x in t[:max_len-1]] for t in df[1]]
	return text, label
(train_text, train_label), (test_text, test_label) = map(convert_data, xys)

# must post padding!
pad_func = lambda x:np.expand_dims(sequence.pad_sequences(x, maxlen=max_len, padding='post', truncating='post'), -1)
train_label, test_label = map(pad_func, [train_label, test_label])

sys.path.append('../')
import bert
train_inputs, test_inputs = map(lambda x:bert.convert_sentences(x, max_len), [train_text, test_text])

bert_inputs = bert.get_bert_inputs(max_len)
bert_output = bert.BERTLayer(n_fine_tune_vars=3, return_sequences=True)(bert_inputs)
x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.25))(bert_output)
crf = CRF(len(y2id), sparse_target=True)
pred = crf(x)
model = Model(inputs=bert_inputs, outputs=pred)

epochs = 2
batch_size = 64
total_steps = epochs*train_inputs[0].shape[0]//batch_size
lr_scheduler, optimizer = bert.get_suggested_scheduler_and_optimizer(init_lr=1e-3, total_steps=total_steps)

model.compile(optimizer, loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

from seqeval.metrics import f1_score, accuracy_score, classification_report
class TestCallback(Callback):
	def __init__(self, XY, model, tags):
		self.X, self.Y = XY
		self.Y = np.squeeze(self.Y, -1)
		self.smodel = model
		self.tags = tags
		self.best_f1 = 0
	def on_epoch_end(self, epoch, logs = None):
		# self.model is auto set by keras
		yt, yp = [], []
		pred =  np.argmax(self.smodel.predict(self.X, batch_size=32), -1)
		lengths = [x.sum() for x in self.X[1]]
		for pseq, yseq, llen in zip(pred, self.Y, lengths):
			yt.append([self.tags[z] for z in pseq[1:llen-1]])
			yp.append([self.tags[z] for z in yseq[1:llen-1]])
		f1 = f1_score(yt, yp)
		self.best_f1 = max(self.best_f1, f1)
		accu = accuracy_score(yt, yp)
		print('\naccu: %.4f  F1: %.4f  BestF1: %.4f\n' % (accu, f1, self.best_f1))
		print(classification_report(yt, yp))

test_cb = TestCallback((test_inputs, test_label), model, id2y)
model.fit(train_inputs, train_label, epochs=epochs, batch_size=batch_size,
		  validation_data=(test_inputs, test_label), callbacks=[test_cb, lr_scheduler])

Y = model.predict_on_batch([x[:8] for x in test_inputs]).argmax(-1)
for ii in range(8):
	tlist = [id2y[x] for x in Y[ii][1:]]
	print(' '.join(['%s/%s'%x for x in zip(test_text[ii].split(), tlist)]))

'''
Epoch 1/2
50658/50658 [==============================] - 463s 9ms/step - loss: 0.0346 - acc: 0.9886 - val_loss: 0.0094 - val_acc: 0.9957

accu: 0.9886  F1: 0.8876  BestF1: 0.8876

             precision    recall  f1-score   support

        LOC       0.91      0.88      0.89      2986
        PER       0.94      0.92      0.93      2026
        ORG       0.82      0.80      0.81      1372

avg / total       0.90      0.87      0.89      6384

Epoch 2/2
50658/50658 [==============================] - 460s 9ms/step - loss: 0.0068 - acc: 0.9958 - val_loss: 0.0047 - val_acc: 0.9962

accu: 0.9898  F1: 0.9041  BestF1: 0.9041

             precision    recall  f1-score   support

        LOC       0.91      0.92      0.91      2834
        PER       0.94      0.93      0.93      1997
        ORG       0.84      0.85      0.84      1326

avg / total       0.90      0.91      0.90      6157
'''
