import sys
sys.path.append('../')

from config import data_config as config
from config.model_config import model_dict2, model_info
from model.sklearn_classifier import SklearnTokenizer
from datasets_prep.dataset import create_test_data_sklearn, prepare_data_sklearn
from fidelity.utility import compute_fidelity
import os
import pickle
import numpy as np
import pandas as pd

OUTPUT_DIR = "/content/human-rationales/output"
DATASET = config.dataset

if __name__ == '__main__':
    for model_name in model_dict2['model']:
        tunable_model_args = model_info[model_name]["tunable_model_args"]
        dset = DATASET
        model_save_path = os.path.join(OUTPUT_DIR, model_name)
        output_dir = os.path.join(OUTPUT_DIR, os.path.join(model_name, 'output'))

        train_path = dset['train_path']
        test_path = dset['test_path']

        tokenizer = SklearnTokenizer(max_length=dset["max_len"])
        train_df, test_df = prepare_data_sklearn(tokenizer=tokenizer, train_path=train_path, test_path=test_path)
        candidate_model = model_name["class"](train_df, dset["max_len"], **tunable_model_args)

        candidate_model.train(train_df=train_df)
        candidate_model.save_model(save_path=model_save_path)

        test_df = create_test_data_sklearn(tokenizer, filepath=dset["test_path"],
													  classes=dset["classes"])

        cache_model = pickle.load(open(os.path.join(model_save_path, "model.sav"), 'rb'))

        # Results
        predicted_classes, prob_y_hat = cache_model.predict(input_ids=test_df["input_ids"])
        prob_y_hat = prob_y_hat[np.arange(len(prob_y_hat)), predicted_classes]
        _, prob_y_hat_alpha = cache_model.predict(input_ids=test_df["sufficiency_input_ids"])
        prob_y_hat_alpha = prob_y_hat_alpha[np.arange(len(prob_y_hat_alpha)), predicted_classes]
        _, prob_y_hat_alpha_comp = cache_model.predict(input_ids=test_df["comprehensiveness_input_ids"])
        prob_y_hat_alpha_comp = prob_y_hat_alpha_comp[np.arange(len(prob_y_hat_alpha_comp)), predicted_classes]
        _, prob_y_hat_0 = cache_model.predict(input_ids=test_df["null_diff_input_ids"])
        prob_y_hat_0_predicted_class = prob_y_hat_0[np.arange(len(prob_y_hat_0)), predicted_classes]
        null_diff = 1 - compute_fidelity(prob_y_hat=prob_y_hat,
                                         prob_y_hat_alpha=prob_y_hat_0_predicted_class,
                                         fidelity_type="sufficiency")

        feature_cache_df = pd.DataFrame({
            'prob_y_hat': prob_y_hat,
            'prob_y_hat_alpha': prob_y_hat_alpha,
            'prob_y_hat_alpha_comp': prob_y_hat_alpha_comp,
            'null_diff': null_diff,
            'true_classes': test_df['labels'],
            'predicted_classes': predicted_classes
        })
        feature_cache_df.to_csv(model_save_path + "/feature.csv")



