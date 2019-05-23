# BERT_keras
An easy-to-use BERT in keras via tf-hub.

## Getting Started

At first, set env variable TFHUB_CACHE_DIR and/or https_proxy for saving pre-train models.

We must use the original tokenizer to be compatible with the pre-train weights.

And the original optimizer leads to better fine-tuning results.

```
import bert
bert.set_language('en')   # or 'cn'
bert_inputs = bert.get_bert_inputs(max_seq_length=128)
bert_output = bert.BERTLayer(n_fine_tune_vars=3, return_sequences=False)(bert_inputs)
x = Dense(1, activation='sigmoid')(bert_output)
model = Model(inputs=bert_inputs, outputs=x)

X = bert.convert_sentences(sentences)   #  X = [input_ids, input_masks, segment_ids]

lr_scheduler, optimizer = bert.get_suggested_scheduler_and_optimizer(init_lr=1e-3, total_steps=total_batchs)
model.compile(optimizer=optimizer, loss=...)
model.fit(X, Y, ..., callbacks=[lr_scheduler])
```

More details are in the examples.

### Some useful functions 

bert.restore_token_list is used to generate a tokenized result that can map to the original sentences.

It is useful in sequence tasks.
```
sent = '@@@ I    have 10 RTX 2080Ti.'
tokens = tokenize_sentence(sent)
# ['@', '@', '@', '[UNK]', 'have', '10', '[UNK]', '[UNK]', '.']
otokens = restore_token_list(sent, tokens)
# ['@', '@', '@', ' I    ', 'have ', '10', ' RTX', ' 2080Ti', '.']
```

## Running the tests

Just run *.py in examples/

## Acknowledgments

The tokenizer and optimizer are from https://github.com/google-research/bert


The baisc BERTLayer is from:
* https://towardsdatascience.com/bert-in-keras-with-tensorflow-hub-76bcbc9417b
* https://github.com/strongio/keras-bert
