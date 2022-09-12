import sys
sys.path.append('../')
from transformers import Trainer, TrainingArguments, EvalPrediction, PretrainedConfig
from typing import Callable, Dict
import sklearn.metrics as mt
import numpy as np

from config.data_config import dataset
from datasets_prep.dataset import prepare_data
from datasets_prep.dataset import create_test_dataloader
from config.trainer_config import training_args_config
from config.model_config import model_dict1, model_info
from train_eval.feature_caching import get_and_save_features

import os
import torch
import json


def build_compute_metrics_fn() -> Callable[[EvalPrediction], Dict]:
	def compute_metrics_fn(p: EvalPrediction):
		preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
		preds = np.argmax(preds, axis=1)
		return {"acc": mt.accuracy_score(p.label_ids, preds)}

	return compute_metrics_fn

OUTPUT_DIR = "../outputs"
DATASET = dataset


if __name__ == '__main__':
    for model_name in model_dict1['model']:
        tunable_model_args = model_info[model_name]["tunable_model_args"]
        dset = DATASET
        model_save_path = os.path.join(OUTPUT_DIR, model_name)

        model_config = PretrainedConfig(
			max_length=dset["max_len"],
			num_labels=len(dset["classes"]),
			**tunable_model_args)

        candidate_model = model_info[model_name]["class"](config=model_config)

        train_path = dset['train_path']
        eval_path = dset['test_path']
        classes = dset['classes']

        print("preparing data")

        train_dataset, eval_dataset = prepare_data(
			model=candidate_model,
			classes=classes,
            train_path=train_path,
            eval_path=eval_path
        )
        
        training_args_config["per_device_train_batch_size"] = dset["batch_size"]
        save_steps = len(train_dataset) // training_args_config['per_device_train_batch_size']

        output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, 'trainingargs'))
        print("training arguments")
        training_args = TrainingArguments(
			output_dir=output_dir,
			save_steps=save_steps,
			**training_args_config,
            gradient_accumulation_steps=5,
            gradient_checkpointing=True
        )

        print("training")
        trainer = Trainer(
			model=candidate_model,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=eval_dataset,
			compute_metrics=build_compute_metrics_fn(),
		)

        trainer.train()
        trainer.save_model(output_dir=model_save_path)

        print("preparing test")
        # Feature Analysis
        test_dataloader = create_test_dataloader(
			model=candidate_model,
			filepath=eval_path,
			classes=classes,
			batch_size=training_args_config["per_device_eval_batch_size"]
		)

        model_load_path = os.path.join(model_save_path, 'pytorch_model.bin')
        with open(os.path.join(model_save_path, 'config.json'), 'r')as f:
            saved_config = json.load(f)
            saved_config = PretrainedConfig(
                num_labels=len(classes),
                **saved_config
            )

        cache_model = model_info[model_name]["class"](config=saved_config)
        cache_model.load_state_dict(torch.load(model_load_path))

        get_and_save_features(
			test_dataloader=test_dataloader,
			model=cache_model,
			tokenizer=cache_model.tokenizer,
			save_dir=os.path.join(OUTPUT_DIR, model_name),
		)


