import torch
import pandas as pd
import json
import sys

sys.path.append('../')
from boolQ.make_csv_data import get_data

class Dataset(torch.utils.data.Dataset):
	def __init__(self, X, labels, attention_masks, BATCH_SIZE_FLAG=32):
		"""Initialization"""
		self.y = labels
		self.X = X
		self.attention_masks = attention_masks
		self.BATCH_SIZE_FLAG = BATCH_SIZE_FLAG

	def __len__(self):
		"""number of samples"""
		return self.X.shape[0]

	def __getitem__(self, index):
		"""Get individual item from the tensor"""
		sample = {"input_ids": self.X[index],
				  "labels": self.y[index],
				  "attention_mask": self.attention_masks[index]
				  }
		return sample

def prepare_data(model, classes, train_path, eval_path, test_path=None, batch_size=32, max_rows=None, max_len=512, return_dataset=False, name=None):
	"""Preparing data for training, evaluation and testing"""

	train_dataloader = create_dataloader(model, classes, train_path, max_rows=max_rows, batch_size=batch_size, max_len=max_len, return_dataset=return_dataset, name=name)
	eval_dataloader = create_dataloader(model, classes, eval_path, max_rows=max_rows, batch_size=batch_size, max_len=max_len, return_dataset=return_dataset, name=name)

	return train_dataloader, eval_dataloader

def create_dataloader(model, classes, filepath, batch_size=32, max_rows=None, class_specific=None, max_len=512, return_dataset=False, name=None):
	data_df = get_data(filepath)
	data_df = data_df[data_df['text'].notna()]
	data_df.reset_index(drop=True, inplace=True)

	try:
		data_df = data_df[data_df['evidences'].notna()]
		data_df.reset_index(drop=True, inplace=True)
	except Exception as e:
		pass

	if max_rows is not None:
		data_df = data_df.iloc[:max_rows]

	data_df['text']= data_df['text'].apply(lambda t:t.replace('[SEP]',model.tokenizer.sep_token))

	data_df['input_ids'], data_df['attention_mask'] = zip(*data_df['text'].map(model.tokenize))

	input_id_tensor = torch.tensor(data_df['input_ids'])
	attention_mask_tensor = torch.tensor(data_df['attention_mask'])

	labels_tensor = create_label_tensor(data_df, classes)
	if class_specific is not None:
		pass

	dataset_ds = Dataset(input_id_tensor, labels_tensor, attention_mask_tensor,
						 BATCH_SIZE_FLAG=batch_size)

	return dataset_ds
    #torch.utils.data.DataLoader(dataset_ds, batch_size=dataset_ds.BATCH_SIZE_FLAG, shuffle=True)

def create_label_tensor(data_df, classes):
	return torch.tensor(data_df['classification'].apply(lambda x: classes.index(x)))


# Testing
class TestDataset(torch.utils.data.Dataset):
	def __init__(
			self, id, input_ids, attention_mask, sufficiency_input_ids, sufficiency_attention_mask,
			comprehensiveness_input_ids, comprehensiveness_attention_mask, labels, batch_size=32):
		"""Initialization"""
		self.id = id
		self.input_ids = input_ids
		self.attention_mask = attention_mask
		self.sufficiency_input_ids = sufficiency_input_ids
		self.sufficiency_attention_mask = sufficiency_attention_mask
		self.comprehensiveness_input_ids = comprehensiveness_input_ids
		self.comprehensiveness_attention_mask = comprehensiveness_attention_mask
		self.labels = labels
		self.batch_size = batch_size

	def __len__(self):
		"""number of samples"""
		return self.input_ids.shape[0]

	def __getitem__(self, index):
		"""Get individual item from the tensor"""
		sample = {
			"id": self.id[index],
			"input_ids": self.input_ids[index],
			"attention_mask": self.attention_mask[index],
			"sufficiency_input_ids": self.sufficiency_input_ids[index],
			"sufficiency_attention_mask": self.sufficiency_attention_mask[index],
			"comprehensiveness_input_ids": self.comprehensiveness_input_ids[index],
			"comprehensiveness_attention_mask": self.comprehensiveness_attention_mask[index],
			"labels": self.labels[index]
		}
		return sample

def create_test_dataloader(model, filepath, classes, batch_size=32):
	data_df = get_data(filepath)

	if "evidences" not in data_df.columns:
		data_df["evidences"] = data_df["text"].apply(lambda s: s.strip("[").strip("]").split())

	data_df = data_df[data_df['evidences'].notna()]
	data_df.reset_index(drop=True, inplace=True)
	data_df["evidences"] = data_df['evidences'].apply(lambda s: json.loads(s))


	data_df["sufficiency_text"] = data_df[
		["text", "evidences"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="sufficiency"), axis=1)
	data_df["comprehensiveness_text"] = data_df[
		["text", "evidences"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="comprehensiveness"), axis=1)

	data_df['sufficiency_input_ids'], data_df['sufficiency_attention_mask'] =\
		zip(*data_df['sufficiency_text'].map(model.tokenize))
	data_df['comprehensiveness_input_ids'], data_df['comprehensiveness_attention_mask'] =\
		zip(*data_df['comprehensiveness_text'].map(model.tokenize))
	data_df['input_ids'], data_df['attention_mask'] = \
		zip(*data_df['text'].map(model.tokenize))

	input_id_tensor = torch.tensor(data_df['input_ids'])
	attention_mask_tensor = torch.tensor(data_df['attention_mask'])

	sufficiency_input_id_tensor = torch.tensor(data_df['sufficiency_input_ids'])
	sufficiency_attention_mask_tensor = torch.tensor(data_df['sufficiency_attention_mask'])

	comprehensiveness_input_id_tensor = torch.tensor(data_df['comprehensiveness_input_ids'])
	comprehensiveness_attention_mask_tensor = torch.tensor(data_df['comprehensiveness_attention_mask'])

	labels_tensor = create_label_tensor(data_df, classes)

	test_dataset_ds = TestDataset(
		id=data_df["annotation_id"],
		input_ids=input_id_tensor,
		attention_mask=attention_mask_tensor,
		sufficiency_input_ids=sufficiency_input_id_tensor,
		sufficiency_attention_mask=sufficiency_attention_mask_tensor,
		comprehensiveness_input_ids=comprehensiveness_input_id_tensor,
		comprehensiveness_attention_mask=comprehensiveness_attention_mask_tensor,
		labels=labels_tensor,
		batch_size=batch_size
	)

	test_dataloader = torch.utils.data.DataLoader(
		test_dataset_ds, batch_size=test_dataset_ds.batch_size, shuffle=True)

	return test_dataloader



def reduce_by_alpha(text, rationale, fidelity_type):
    reduced_text = ""
    tokens = text.split()
    evidences=evidences.lower()

    for token in tokens:
        token = token.lower()
        try:
            if fidelity_type=="sufficiency" and contains_word(rationale, token):
                reduced_text = reduced_text + token + " "
            elif fidelity_type=="comprehensiveness" and not contains_word(evidences, token):
                reduced_text = reduced_text + token + " "
        except Exception as e:
            if fidelity_type == "comprehensiveness":
                reduced_text = reduced_text + token + " "
    
    if len(reduced_text) > 0:
        reduced_text = reduced_text[:-1]

    return reduced_text

def contains_word(sentence, word):
    return (' ' + word + ' ') in (' ' + sentence + ' ')

# Sklearn
def prepare_data_sklearn(tokenizer, train_path, test_path, classes=None):
	train_df = create_tokenized_data(tokenizer, train_path, classes)
	test_df = create_tokenized_data(tokenizer, test_path, classes)
	return train_df, test_df

def create_tokenized_data(tokenizer, filepath, classes):
	data_df = get_data(filepath)
	data_df['input_ids'], data_df['attention_mask'] = zip(*data_df['text'].map(tokenizer.tokenize))
	data_df["labels"] = data_df['classification'].apply(lambda x: classes.index(x))
	return data_df

def create_test_data_sklearn(tokenizer, filepath, classes):
	"""preparing the test dataloader"""
	data_df = get_data(filepath)

	data_df = data_df[data_df['evidences'].notna()]
	data_df.reset_index(drop=True, inplace=True)
	data_df["evidences"] = data_df['evidences'].apply(lambda s: json.loads(s))

	data_df["sufficiency_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="sufficiency"), axis=1)
	data_df["comprehensiveness_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="comprehensiveness"), axis=1)
	data_df["null_diff_text"] = data_df[
		["text", "rationale"]].apply(lambda s: reduce_by_alpha(*s, fidelity_type="null_diff"), axis=1)

	data_df['sufficiency_input_ids'], data_df['sufficiency_attention_mask'] =\
		zip(*data_df['sufficiency_text'].map(tokenizer.tokenize))
	data_df['comprehensiveness_input_ids'], data_df['comprehensiveness_attention_mask'] =\
		zip(*data_df['comprehensiveness_text'].map(tokenizer.tokenize))
	data_df['null_diff_input_ids'], data_df['null_diff_attention_mask'] = \
		zip(*data_df['null_diff_text'].map(tokenizer.tokenize))

	data_df['input_ids'], data_df['attention_mask'] = \
		zip(*data_df['text'].map(tokenizer.tokenize))

	data_df["labels"] = data_df['classification'].apply(lambda x: classes.index(x))

	return data_df
