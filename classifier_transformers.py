# Basic baseline classifier for toxic spans detection
# File name: classifier_baseline.py
# Date: 08-10-2020

# Suppress warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import
import pyconll
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from seqeval.metrics import f1_score, accuracy_score
import transformers
from transformers import BertTokenizer, BertConfig
from transformers import get_linear_schedule_with_warmup
from transformers import BertForTokenClassification, AdamW
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import csv
import re
import numpy as np


def read_data(data_file):
	conll = pyconll.load_from_file(data_file)
	sentence_list = []
	label_list = []
	for sentence in conll:
		current_token_list, current_label_list = [], []
		for token in sentence:
			current_token_list.append(token.form)
			current_label_list.append(int(list(token.misc.keys())[0]))
		sentence_list.append(current_token_list)
		label_list.append(current_label_list)

	return sentence_list, label_list


def read_predict_file(predict_file):
	sentence_list, label_list = [], []
	with open(predict_file, "r", encoding='utf-8') as f:
		data = csv.reader(f, delimiter=',', quotechar='"')
		for row in data:
			label_list.append(row[0])
			sentence_list.append(row[1])

	return sentence_list, label_list


def tokenize_and_preserve_labels(sentence, text_labels, tokenizer):
	tokenized_sentence = []
	labels = []

	for word, label in zip(sentence, text_labels):

		# Tokenize the word and count # of subwords the word is broken into
		tokenized_word = tokenizer.tokenize(word)
		n_subwords = len(tokenized_word)

		# Add the tokenized word to the final tokenized word list
		tokenized_sentence.extend(tokenized_word)

		# Add the same label to the new list of labels `n_subwords` times
		labels.extend([label] * n_subwords)

	return tokenized_sentence, labels


def process_data(sentence_list, label_list, tokenizer):
	# Parameters
	max_seq_length = 75
	batch_size = 32

	tokenized_texts_and_labels = [
		tokenize_and_preserve_labels(sent, labs, tokenizer)
		for sent, labs in zip(sentence_list, label_list)
	]

	tokenized_text_list = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
	tokenized_label_list = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

	input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_text_list], maxlen=max_seq_length, dtype="long", value=0.0, truncating="post", padding="post")
	input_tags = pad_sequences(tokenized_label_list, maxlen=max_seq_length, value=2, padding="post", dtype="long", truncating="post")

	attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

	tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, input_tags, random_state=2018, test_size=0.1)
	tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

	tr_inputs = torch.tensor(tr_inputs)
	val_inputs = torch.tensor(val_inputs)
	tr_tags = torch.tensor(tr_tags)
	val_tags = torch.tensor(val_tags)
	tr_masks = torch.tensor(tr_masks)
	val_masks = torch.tensor(val_masks)

	train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

	valid_data = TensorDataset(val_inputs, val_masks, val_tags)
	valid_sampler = SequentialSampler(valid_data)
	valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size)

	return train_dataloader, valid_dataloader


def create_model(size_train_dataloader):
	model = BertForTokenClassification.from_pretrained(
		"bert-base-cased",
		num_labels=3,
		output_attentions=False,
		output_hidden_states=False
	)

	model.cuda()

	FULL_FINETUNING = True
	if FULL_FINETUNING:
		param_optimizer = list(model.named_parameters())
		no_decay = ['bias', 'gamma', 'beta']
		optimizer_grouped_parameters = [
			{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
			 'weight_decay_rate': 0.01},
			{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
			 'weight_decay_rate': 0.0}
		]
	else:
		param_optimizer = list(model.classifier.named_parameters())
		optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

	optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

	epochs = 3
	max_grad_norm = 1.0

	# Total number of training steps is number of batches * number of epochs.
	total_steps = size_train_dataloader * epochs

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(
		optimizer,
		num_warmup_steps=0,
		num_training_steps=total_steps
	)

	return model, scheduler, optimizer


def train_model(model, scheduler, optimizer, train_dataloader, valid_dataloader, device, epochs, max_grad_norm):
	loss_values, validation_loss_values = [], []

	for _ in trange(epochs, desc="Epoch"):
		# Put the model into training mode.
		model.train()
		# Reset the total loss for this epoch.
		total_loss = 0

		# Training loop
		for step, batch in enumerate(train_dataloader):
			# add batch to gpu
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch

			b_input_ids = torch.tensor(b_input_ids).to(torch.int64)  # Quick fix for windows
			b_input_mask = torch.tensor(b_input_mask).to(torch.int64)  # Quick fix for windows
			b_labels = torch.tensor(b_labels).to(torch.int64)  # Quick fix for windows

			# Always clear any previously calculated gradients before performing a backward pass.
			model.zero_grad()
			# forward pass
			# This will return the loss (rather than the model output)
			# because we have provided the `labels`.
			outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
			# get the loss
			loss = outputs[0]
			# Perform a backward pass to calculate the gradients.
			loss.backward()
			# track train loss
			total_loss += loss.item()
			# Clip the norm of the gradient
			# This is to help prevent the "exploding gradients" problem.
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
			# update parameters
			optimizer.step()
			# Update the learning rate.
			scheduler.step()

		# Calculate the average loss over the training data.
		avg_train_loss = total_loss / len(train_dataloader)
		print("Average train loss: {}".format(avg_train_loss))

		# Store the loss value for plotting the learning curve.
		loss_values.append(avg_train_loss)

		# After the completion of each training epoch, measure our performance on
		# our validation set.

		# Put the model into evaluation mode
		model.eval()
		# Reset the validation loss for this epoch.
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		predictions, true_labels = [], []
		for batch in valid_dataloader:
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch

			b_input_ids = torch.tensor(b_input_ids).to(torch.int64)  # Quick fix for windows
			b_input_mask = torch.tensor(b_input_mask).to(torch.int64)  # Quick fix for windows
			b_labels = torch.tensor(b_labels).to(torch.int64)  # Quick fix for windows

			# Telling the model not to compute or store gradients,
			# saving memory and speeding up validation
			with torch.no_grad():
				# Forward pass, calculate logit predictions.
				# This will return the logits rather than the loss because we have not provided labels.
				outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
			# Move logits and labels to CPU
			logits = outputs[1].detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()

			# Calculate the accuracy for this batch of test sentences.
			eval_loss += outputs[0].mean().item()
			predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
			true_labels.extend(label_ids)

		eval_loss = eval_loss / len(valid_dataloader)
		validation_loss_values.append(eval_loss)
		print("Validation loss: {}".format(eval_loss))

		pred_tags = [p_i for p, l in zip(predictions, true_labels)
					 for p_i, l_i in zip(p, l) if l_i != 2]
		valid_tags = [l_i for l in true_labels
					  for l_i in l if l_i != 2]

		#print(type(pred_tags))
		#print(type(valid_tags))



		#print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
		#print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
	return model


def predict_sentence(model, tokenizer, predict_sentence):
	tokenized_sentence = tokenizer.encode(predict_sentence)
	input_ids = torch.tensor([tokenized_sentence]).cuda()

	with torch.no_grad():
		output = model(input_ids)
	label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

	# join bpe split tokens
	tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
	new_tokens, new_labels = [], []
	for token, label_idx in zip(tokens, label_indices[0]):
		if token.startswith("##"):
			new_tokens[-1] = new_tokens[-1] + token[2:]
		else:
			new_labels.append(label_idx)
			new_tokens.append(token)

	# retrieve spans of toxic tokens
	result_spans = set()
	for token, label in zip(new_tokens, new_labels):
		token_escape = re.escape(token)
		p = re.compile(token_escape)
		if label == 1:
			for m in p.finditer(predict_sentence):
				if m is not None:
					for i in range(m.start(),m.end()):
						result_spans.add(i)

	return [list(result_spans), predict_sentence]


def write_results(results, filename):
	with open(filename, "w", encoding='utf-8') as f:
		for result in results:
			result_string = '"' + str(result[0]) + '","' + str(result[1]) + '"\n'
			f.write(result_string)


def main(train=False, predict_file=None):
	# Parameters
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()
	epochs = 4
	max_grad_norm = 1.0

	# Tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)

	# Read data
	sentence_list, label_list = read_data("Data/converted_data_train.conll")

	# Process data
	train_dataloader, valid_dataloader = process_data(sentence_list, label_list, tokenizer)

	# Train if required
	if train:
		model, scheduler, optimizer = create_model(len(train_dataloader))
		train_model(model, scheduler, optimizer, train_dataloader, valid_dataloader, device, epochs, max_grad_norm)
		torch.save(model, "toxic_classifier.model")
	else:
		model = torch.load("toxic_classifier.model")

	# Predict
	sentence_list, label_list = read_predict_file("Data/tsd_train.csv")
	results = []
	for sentence in sentence_list:
		result = predict_sentence(model, tokenizer, sentence)
		results.append(result)

	write_results(results, "results.csv")


if __name__ == "__main__":
	train_mode = False
	if len(sys.argv) > 1:
		train_mode = sys.argv[1].lower() == "true"

	predict_file = None
	if len(sys.argv) > 2:
		predict_file = sys.argv[2]

	main(train_mode, predict_file)