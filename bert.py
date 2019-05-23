'''
The basic idea is from https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
'''
import os, re, sys
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.layers import *
from keras.models import *
import numpy as np

if "TFHUB_CACHE_DIR" not in os.environ: 
	os.environ["TFHUB_CACHE_DIR"] = os.path.join(os.path.expanduser('~'), 'tfhub')
if "https_proxy" not in os.environ: 
	if not os.path.exists(os.environ["TFHUB_CACHE_DIR"]):
		print('Warning: You may need to set up an https proxy to download the pretrain model from Google.')
	os.environ["https_proxy"] = ""

cn_bert_path = "https://tfhub.dev/google/bert_chinese_L-12_H-768_A-12/1"
en_bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
bert_path = cn_bert_path

def set_language(lang='cn'):
	global bert_path
	if lang not in ['en', 'cn']: print('Please set_language with en/cn')
	bert_path = cn_bert_path if lang == 'cn' else en_bert_path
	
sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())
K.set_session(sess)

# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

from tokenization import FullTokenizer
def create_tokenizer_from_hub_module():
	global do_lower_case
	"""Get the vocab file and casing info from the Hub module."""
	bert_module = hub.Module(bert_path)
	tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
	vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"], tokenization_info["do_lower_case"]])
	return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
tokenizer = None

def convert_single_sentence(token_list, max_seq_length):
	tokens = ["[CLS]"]
	tokens += token_list[:max_seq_length-2]
	tokens.append("[SEP]") 
	return tokens


def convert_sentences(sents, max_seq_length=256):
	global tokenizer
	if tokenizer is None: tokenizer = create_tokenizer_from_hub_module()
	from tqdm import tqdm
	shape = (len(sents), max_seq_length)
	input_ids, input_masks, segment_ids = map(lambda x:np.zeros(shape, dtype='int32'), '123')
	ii = 0
	for sent in tqdm(sents, desc="Converting sentences"):
		token_list = tokenizer.tokenize(sent)
		tokens = convert_single_sentence(token_list, max_seq_length)
		idlist = tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
		input_masks[ii,:len(idlist)] = 1
		ii += 1
	return [input_ids, input_masks, segment_ids]

def tokenize_sentence(sent):
	global tokenizer
	if tokenizer is None: tokenizer = create_tokenizer_from_hub_module()
	return tokenizer.tokenize(sent)

def convert_tokens(token_lists, max_seq_length=256):
	shape = (len(token_lists), max_seq_length)
	input_ids, input_masks, segment_ids = map(lambda x:np.zeros(shape, dtype='int32'), '123')
	for ii, token_list in enumerate(token_lists):
		tokens = convert_single_sentence(token_list, max_seq_length)
		idlist = tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
		input_masks[ii,:len(idlist)] = 1
	return [input_ids, input_masks, segment_ids]

def convert_pair_tokens(tokens_ab, max_seq_length=256, max_a_length=256):
	shape = (len(tokens_ab), max_seq_length)
	input_ids, input_masks, segment_ids = map(lambda x:np.zeros(shape, dtype='int32'), '123')
	for ii, (ta, tb) in enumerate(tokens_ab):
		tokens = ['[CLS]'] + ta[:max_a_length] + ['[SEP]']
		alen = len(tokens)
		max_b_len = max_seq_length - alen - 1
		tokens += tb[:max_b_len] + ['[SEP]']
		idlist = tokenizer.convert_tokens_to_ids(tokens)
		input_ids[ii,:len(idlist)] = idlist
		input_masks[ii,:len(idlist)] = 1
		segment_ids[ii,alen:len(idlist)] = 1
	return [input_ids, input_masks, segment_ids]

class BERTLayer(Layer):
	def __init__(self, n_fine_tune_vars=10, return_sequences=False, **kwargs):
		super().__init__(**kwargs)
		self.return_sequences = return_sequences
		self.output_key = 'sequence_output' if return_sequences else 'pooled_output'
		self.n_fine_tune_vars = n_fine_tune_vars
		self.output_size = 768
	def build(self, input_shape):
		trainable = self.n_fine_tune_vars > 0
		self.bert = hub.Module(bert_path, trainable=trainable, name="{}_module".format(self.name))
		trainable_vars = self.bert.variables
		
		# Remove unused layers
		trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]

		def pri(v):
			vname = v.name
			for i in range(10, 26): vname = vname.replace('layer_%d'%i, 'layer_%s'%chr(55+i))
			return vname
		#trainable_vars.sort(key=pri)
		# No sort is better, why? It fine-tunes layer 9 first.

		if self.return_sequences: trainable_vars = trainable_vars[:-2] 
		trainable_vars = trainable_vars[-self.n_fine_tune_vars:] if self.n_fine_tune_vars > 0 else []

		# Add to trainable weights
		for var in trainable_vars:
			self._trainable_weights.append(var)
		for var in self.bert.variables:
			if var not in self._trainable_weights:
				self._non_trainable_weights.append(var)
		super().build(input_shape)

	def call(self, inputs):
		inputs = [K.cast(x, dtype="int32") for x in inputs]
		input_ids, input_mask, segment_ids = inputs
		bert_inputs = dict(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
		result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[self.output_key]
		return result

	def compute_output_shape(self, input_shape):
		if self.return_sequences: return (input_shape[0][0], input_shape[0][1], self.output_size)
		return (input_shape[0][0], self.output_size)

def get_bert_inputs(max_seq_length, name_prefix='input'):
	in_id =      Input(shape=(max_seq_length,), name=name_prefix+"bert_ids")
	in_mask =    Input(shape=(max_seq_length,), name=name_prefix+"bert_masks")
	in_segment = Input(shape=(max_seq_length,), name=name_prefix+"bert_segids")
	bert_inputs = [in_id, in_mask, in_segment]
	return bert_inputs

def get_suggested_scheduler(init_lr=5e-5, total_steps=10000, warmup_ratio=0.1):
	opt_lr = K.variable(init_lr)
	warmup_steps = int(warmup_ratio * total_steps)
	warmup = WarmupCallback(opt_lr, init_lr, total_steps, warmup_steps)
	return warmup, opt_lr

def get_suggested_optimizer(opt_lr):
	optimizer = AdamWeightDecayOptimizer(learning_rate=opt_lr, 
				weight_decay_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-6,
				exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
	return optimizer

def get_suggested_scheduler_and_optimizer(init_lr=5e-5, total_steps=10000, warmup_ratio=0.1):
	warmup, opt_lr = get_suggested_scheduler(init_lr, total_steps, warmup_ratio)
	return warmup, get_suggested_optimizer(opt_lr)

from keras.callbacks import Callback
class WarmupCallback(Callback):
	def __init__(self, lr_var, init_lr, total_steps, warmup_steps=0):
		self.step = 0
		self.lr_var = lr_var
		self.init_lr = init_lr
		self.warmup = warmup_steps
		self.total_steps = total_steps
	def on_batch_begin(self, batch, logs):
		self.step += 1
		if self.step <= self.warmup: 
			new_lr = self.init_lr * (self.step / self.warmup)
		else: 
			new_lr = self.init_lr * max(0, 1 - self.step / self.total_steps)
		K.set_value(self.lr_var, new_lr)


class AdamWeightDecayOptimizer(tf.train.Optimizer):
	"""A basic Adam optimizer that includes "correct" L2 weight decay."""

	def __init__(self, learning_rate, weight_decay_rate=0.0,
				beta_1=0.9, beta_2=0.999, epsilon=1e-6, exclude_from_weight_decay=None,
				name="AdamWeightDecayOptimizer"):
		"""Constructs a AdamWeightDecayOptimizer."""
		super(AdamWeightDecayOptimizer, self).__init__(False, name)

		self.learning_rate = learning_rate
		self.weight_decay_rate = weight_decay_rate
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.exclude_from_weight_decay = exclude_from_weight_decay

	def apply_gradients(self, grads_and_vars, global_step=None, name=None):
		"""See base class."""
		assignments = []
		grads, tvars = [x[0] for x in grads_and_vars], [x[1] for x in grads_and_vars]
		(grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
		grads_and_vars = zip(grads, tvars)
		for (grad, param) in grads_and_vars:
			if grad is None or param is None: continue

			param_name = self._get_variable_name(param.name)

			m = tf.get_variable(name=param_name + "/adam_m",
					shape=param.shape.as_list(),
					dtype=tf.float32,
					trainable=False,
					initializer=tf.zeros_initializer())
			v = tf.get_variable(name=param_name + "/adam_v",
					shape=param.shape.as_list(),
					dtype=tf.float32,
					trainable=False,
					initializer=tf.zeros_initializer())

			# Standard Adam update.
			next_m = (tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
			next_v = (tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

			update = next_m / (tf.sqrt(next_v) + self.epsilon)

			# Just adding the square of the weights to the loss function is *not*
			# the correct way of using L2 regularization/weight decay with Adam,
			# since that will interact with the m and v parameters in strange ways.
			#
			# Instead we want ot decay the weights in a manner that doesn't interact
			# with the m/v parameters. This is equivalent to adding the square
			# of the weights to the loss with plain (non-momentum) SGD.
			if self._do_use_weight_decay(param_name):
				update += self.weight_decay_rate * param

			update_with_lr = self.learning_rate * update

			next_param = param - update_with_lr

			assignments.extend([param.assign(next_param),
						m.assign(next_m),
						v.assign(next_v)])
		return tf.group(*assignments, name=name)

	def _do_use_weight_decay(self, param_name):
		"""Whether to use L2 weight decay for `param_name`."""
		if not self.weight_decay_rate: return False
		if self.exclude_from_weight_decay:
			for r in self.exclude_from_weight_decay:
				if re.search(r, param_name) is not None:
					return False
		return True

	def _get_variable_name(self, param_name):
		"""Get the variable name from the tensor name."""
		m = re.match("^(.*):\\d+$", param_name)
		if m is not None: param_name = m.group(1)
		return param_name


## THESE FUNCTIONS ARE TESTED FOR CHS LANGUAGE ONLY
def gen_token_list_inv_pointer(sent, token_list):
	if do_lower_case: sent = sent.lower()
	otiis = []; iis = 0 
	for it, token in enumerate(token_list):
		otoken = token.lstrip('#')
		if token[0] == '[' and token[-1] == ']': otoken = ''
		niis = iis
		while niis < len(sent):
			if sent[niis:].startswith(otoken): break
			niis += 1
		otiis.append(niis)
		iis = niis + max(1, len(otoken))
	#for tt, ii in zip(token_list, otiis): print(tt, sent[ii:ii+len(tt)])
	#for i, iis in enumerate(otiis): 
	#	assert iis < len(sent)
	#	otoken = token_list[i].strip('#')
	#	assert otoken == '[UNK]' or sent[iis:iis+len(otoken)] == otoken
	return otiis

# restore [UNK] tokens to the original tokens
def restore_token_list(sent, token_list):
	invp = gen_token_list_inv_pointer(sent, token_list)
	invp.append(len(sent))
	otokens = [sent[u:v] for u,v in zip(invp, invp[1:])]
	processed = -1
	for ii, tk in enumerate(token_list):
		if tk != '[UNK]': continue
		if ii < processed: continue
		for jj in range(ii+1, len(token_list)):
			if token_list[jj] != '[UNK]': break
		else: jj = len(token_list)
		allseg = sent[invp[ii]:invp[jj]]

		if ii + 1 == jj: continue
		seppts = [0] + [i for i, x in enumerate(allseg) if i > 0 and i+1 < len(allseg) and x == ' ' and allseg[i-1] != ' ']
		if allseg[seppts[-1]:].replace(' ', '') == '': seppts = seppts[:-1]
		seppts.append(len(allseg))
		if len(seppts) == jj - ii + 1:
			for k, (u,v) in enumerate(zip(seppts, seppts[1:])): 
				otokens[ii+k] = allseg[u:v]
		processed = jj + 1
	if invp[0] > 0: otokens[0] = sent[:invp[0]] + otokens[0]
	if ''.join(otokens) != sent:
		raise Exception('restore tokens failed, text and restored:\n%s\n%s' % (sent, ''.join(otokens)))
	return otokens

def gen_word_level_labels(sent, token_list, word_list, pos_list=None):
	otiis = gen_token_list_inv_pointer(sent, token_list)
	wdiis = [];	iis = 0
	for ip, pword in enumerate(word_list):
		niis = iis
		while niis < len(sent):
			if pword == '' or sent[niis:].startswith(pword[0]): break
			niis += 1
		wdiis.append(niis)
		iis = niis + len(pword)
	#for tt, ii in zip(word_list, wdiis): print(tt, sent[ii:ii+len(tt)])

	rlist = [];	ip = 0
	for it, iis in enumerate(otiis):
		while ip + 1 < len(wdiis) and wdiis[ip+1] <= iis: ip += 1
		if iis == wdiis[ip]: rr = 'B'
		elif iis > wdiis[ip]: rr = 'I'
		rr += '-' + pos_list[ip]
		rlist.append(rr)
	#for rr, tt in zip(rlist, token_list): print(rr, tt)
	return rlist

if __name__ == '__main__':
	sent = '@@@ I    have 10 RTX 2080Ti.'
	tokens = tokenize_sentence(sent)
	otokens = restore_token_list(sent, tokens)
	print(tokens)
	print(otokens)
	print('done')